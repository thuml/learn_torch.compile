
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
                    auto tmp0 = in_ptr0[static_cast<long>(x0)];
                    auto tmp1 = decltype(tmp0)(tmp0 + 250112);
                    auto tmp2 = tmp0 < 0;
                    auto tmp3 = tmp2 ? tmp1 : tmp0;
                    TORCH_CHECK((0 <= tmp3) & (tmp3 < 250112L), "index out of bounds: 0 <= tmp3 < 250112L")
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp6 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = decltype(tmp1)(tmp1 + 250112);
                auto tmp3 = tmp1 < 0;
                auto tmp4 = tmp3 ? tmp2 : tmp1;
                TORCH_CHECK((0 <= tmp4) & (tmp4 < 250112L), "index out of bounds: 0 <= tmp4 < 250112L")
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
''')


cpp_fused__softmax_1 = async_compile.cpp('''
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
}
''')


cpp_fused__softmax_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (384L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp1 = decltype(tmp0)(tmp0 + 250112);
                    auto tmp2 = tmp0 < 0;
                    auto tmp3 = tmp2 ? tmp1 : tmp0;
                    TORCH_CHECK((0 <= tmp3) & (tmp3 < 250112L), "index out of bounds: 0 <= tmp3 < 250112L")
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*tmp3)));
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp7 = tmp6 * tmp6;
                    auto tmp9 = decltype(tmp8)(tmp8 + 250112);
                    auto tmp10 = tmp8 < 0;
                    auto tmp11 = tmp10 ? tmp9 : tmp8;
                    TORCH_CHECK((0 <= tmp11) & (tmp11 < 250112L), "index out of bounds: 0 <= tmp11 < 250112L")
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp2 = decltype(tmp1)(tmp1 + 250112);
                auto tmp3 = tmp1 < 0;
                auto tmp4 = tmp3 ? tmp2 : tmp1;
                TORCH_CHECK((0 <= tmp4) & (tmp4 < 250112L), "index out of bounds: 0 <= tmp4 < 250112L")
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
                auto tmp19 = decltype(tmp18)(tmp18 + 250112);
                auto tmp20 = tmp18 < 0;
                auto tmp21 = tmp20 ? tmp19 : tmp18;
                TORCH_CHECK((0 <= tmp21) & (tmp21 < 250112L), "index out of bounds: 0 <= tmp21 < 250112L")
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
''')


cpp_fused__softmax_5 = async_compile.cpp('''
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
}
''')


cpp_fused__softmax_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (384L*x0)));
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
                    auto tmp0 = in_ptr0[static_cast<long>(x0)];
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = decltype(tmp0)(tmp0 + 250112);
                    auto tmp2 = tmp0 < 0;
                    auto tmp3 = tmp2 ? tmp1 : tmp0;
                    TORCH_CHECK((0 <= tmp3) & (tmp3 < 250112L), "index out of bounds: 0 <= tmp3 < 250112L")
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                auto tmp8 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = decltype(tmp1)(tmp1 + 250112);
                auto tmp3 = tmp1 < 0;
                auto tmp4 = tmp3 ? tmp2 : tmp1;
                TORCH_CHECK((0 <= tmp4) & (tmp4 < 250112L), "index out of bounds: 0 <= tmp4 < 250112L")
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
''')


cpp_fused_add_mul_pow_tanh_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                tmp19.store(in_out_ptr0 + static_cast<long>(x0));
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
                    auto tmp0 = in_ptr0[static_cast<long>(x0)];
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = decltype(tmp0)(tmp0 + 250112);
                    auto tmp2 = tmp0 < 0;
                    auto tmp3 = tmp2 ? tmp1 : tmp0;
                    TORCH_CHECK((0 <= tmp3) & (tmp3 < 250112L), "index out of bounds: 0 <= tmp3 < 250112L")
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                auto tmp10 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = decltype(tmp1)(tmp1 + 250112);
                auto tmp3 = tmp1 < 0;
                auto tmp4 = tmp3 ? tmp2 : tmp1;
                TORCH_CHECK((0 <= tmp4) & (tmp4 < 250112L), "index out of bounds: 0 <= tmp4 < 250112L")
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
''')


cpp_fused__softmax_11 = async_compile.cpp('''
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
}
''')


cpp_fused__softmax_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (384L*x0)));
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
                    auto tmp0 = in_ptr0[static_cast<long>(x0)];
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = decltype(tmp0)(tmp0 + 250112);
                    auto tmp2 = tmp0 < 0;
                    auto tmp3 = tmp2 ? tmp1 : tmp0;
                    TORCH_CHECK((0 <= tmp3) & (tmp3 < 250112L), "index out of bounds: 0 <= tmp3 < 250112L")
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_pow_tanh_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                tmp19.store(in_out_ptr0 + static_cast<long>(x0));
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
}
''')


cpp_fused__softmax_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (384L*x0)));
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_pow_tanh_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                tmp19.store(in_out_ptr0 + static_cast<long>(x0));
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_23 = async_compile.cpp('''
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
}
''')


cpp_fused__softmax_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (384L*x0)));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_pow_tanh_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                tmp19.store(in_out_ptr0 + static_cast<long>(x0));
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_29 = async_compile.cpp('''
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
}
''')


cpp_fused__softmax_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (384L*x0)));
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_pow_tanh_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                tmp19.store(in_out_ptr0 + static_cast<long>(x0));
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_35 = async_compile.cpp('''
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
}
''')


cpp_fused__softmax_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (384L*x0)));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_pow_tanh_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                tmp19.store(in_out_ptr0 + static_cast<long>(x0));
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
                       float* out_ptr0,
                       float* out_ptr1)
{
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_41 = async_compile.cpp('''
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
}
''')


cpp_fused__softmax_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_pow_tanh_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                tmp19.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_47 = async_compile.cpp('''
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
}
''')


cpp_fused__softmax_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_50 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_pow_tanh_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                tmp19.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_53 = async_compile.cpp('''
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
}
''')


cpp_fused__softmax_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_embedding_mean_mul_pow_rsqrt_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const long* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
                    auto tmp0 = in_ptr0[static_cast<long>(x0)];
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = decltype(tmp0)(tmp0 + 250112);
                    auto tmp2 = tmp0 < 0;
                    auto tmp3 = tmp2 ? tmp1 : tmp0;
                    TORCH_CHECK((0 <= tmp3) & (tmp3 < 250112L), "index out of bounds: 0 <= tmp3 < 250112L")
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                auto tmp10 = out_ptr0[static_cast<long>(x0)];
                auto tmp2 = decltype(tmp1)(tmp1 + 250112);
                auto tmp3 = tmp1 < 0;
                auto tmp4 = tmp3 ? tmp2 : tmp1;
                TORCH_CHECK((0 <= tmp4) & (tmp4 < 250112L), "index out of bounds: 0 <= tmp4 < 250112L")
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
''')


cpp_fused_add_mul_pow_tanh_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                tmp19.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_embedding_mean_mul_pow_rsqrt_58 = async_compile.cpp('''
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
                    auto tmp0 = in_ptr0[static_cast<long>(x0)];
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = decltype(tmp0)(tmp0 + 250112);
                    auto tmp2 = tmp0 < 0;
                    auto tmp3 = tmp2 ? tmp1 : tmp0;
                    TORCH_CHECK((0 <= tmp3) & (tmp3 < 250112L), "index out of bounds: 0 <= tmp3 < 250112L")
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_59 = async_compile.cpp('''
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
}
''')


cpp_fused__softmax_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
}
''')


cpp_fused__softmax_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (384L*x0)));
                }
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_pow_tanh_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                tmp19.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_69 = async_compile.cpp('''
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
}
''')


cpp_fused__softmax_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_72 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_73 = async_compile.cpp('''
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
}
''')


cpp_fused__softmax_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (384L*x0)));
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_pow_tanh_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                tmp19.store(in_out_ptr0 + static_cast<long>(x0));
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_79 = async_compile.cpp('''
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
}
''')


cpp_fused__softmax_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_82 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_83 = async_compile.cpp('''
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
}
''')


cpp_fused__softmax_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_85 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (384L*x0)));
                }
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_pow_tanh_87 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                tmp19.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_88 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_89 = async_compile.cpp('''
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
}
''')


cpp_fused__softmax_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_91 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_93 = async_compile.cpp('''
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
}
''')


cpp_fused__softmax_94 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_95 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_96 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_pow_tanh_97 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                tmp19.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_98 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_99 = async_compile.cpp('''
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
}
''')


cpp_fused__softmax_100 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_101 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_102 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
}
''')


cpp_fused__softmax_104 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_105 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_106 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_pow_tanh_107 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                tmp19.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_108 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_109 = async_compile.cpp('''
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
}
''')


cpp_fused__softmax_110 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_112 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_113 = async_compile.cpp('''
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
}
''')


cpp_fused__softmax_114 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_115 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_116 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_pow_tanh_117 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                tmp19.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_118 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_119 = async_compile.cpp('''
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
}
''')


cpp_fused__softmax_120 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_121 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_122 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__softmax_123 = async_compile.cpp('''
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
}
''')


cpp_fused__softmax_124 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_125 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (8192L*x1)));
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_126 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused_add_mul_pow_tanh_127 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                auto tmp19 = tmp17 * tmp18;
                tmp19.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_mul_pow_rsqrt_128 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    auto in_ptr0 = in_out_ptr0;
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
''')


cpp_fused__log_softmax_nll_loss_forward_129 = async_compile.cpp('''
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
        #pragma omp single
        {
            {
                {
                    float tmp_acc0 = 0;
                    long tmp_acc1 = 0;
                    for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x0)];
                        auto tmp9 = out_ptr0[static_cast<long>(x0)];
                        auto tmp11 = out_ptr1[static_cast<long>(x0)];
                        auto tmp1 = static_cast<long>(-100);
                        auto tmp2 = tmp0 != tmp1;
                        auto tmp3 = static_cast<long>(0);
                        auto tmp4 = tmp2 ? tmp0 : tmp3;
                        auto tmp5 = decltype(tmp4)(tmp4 + 250112);
                        auto tmp6 = tmp4 < 0;
                        auto tmp7 = tmp6 ? tmp5 : tmp4;
                        TORCH_CHECK((0 <= tmp7) & (tmp7 < 250112L), "index out of bounds: 0 <= tmp7 < 250112L")
                        auto tmp8 = in_ptr0[static_cast<long>(tmp7 + (250112L*x0))];
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1 = args
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
    assert_size_stride(arg32_1, (512, ), (1, ))
    assert_size_stride(arg33_1, (512, ), (1, ))
    assert_size_stride(arg34_1, (512, ), (1, ))
    assert_size_stride(arg35_1, (512, ), (1, ))
    assert_size_stride(arg36_1, (512, ), (1, ))
    assert_size_stride(arg37_1, (512, ), (1, ))
    assert_size_stride(arg38_1, (512, ), (1, ))
    assert_size_stride(arg39_1, (512, ), (1, ))
    assert_size_stride(arg40_1, (512, ), (1, ))
    assert_size_stride(arg41_1, (512, ), (1, ))
    assert_size_stride(arg42_1, (250112, 512), (512, 1))
    assert_size_stride(arg43_1, (384, 512), (512, 1))
    assert_size_stride(arg44_1, (384, 512), (512, 1))
    assert_size_stride(arg45_1, (384, 512), (512, 1))
    assert_size_stride(arg46_1, (32, 6), (6, 1))
    assert_size_stride(arg47_1, (512, 384), (384, 1))
    assert_size_stride(arg48_1, (1024, 512), (512, 1))
    assert_size_stride(arg49_1, (1024, 512), (512, 1))
    assert_size_stride(arg50_1, (512, 1024), (1024, 1))
    assert_size_stride(arg51_1, (384, 512), (512, 1))
    assert_size_stride(arg52_1, (384, 512), (512, 1))
    assert_size_stride(arg53_1, (384, 512), (512, 1))
    assert_size_stride(arg54_1, (512, 384), (384, 1))
    assert_size_stride(arg55_1, (1024, 512), (512, 1))
    assert_size_stride(arg56_1, (1024, 512), (512, 1))
    assert_size_stride(arg57_1, (512, 1024), (1024, 1))
    assert_size_stride(arg58_1, (384, 512), (512, 1))
    assert_size_stride(arg59_1, (384, 512), (512, 1))
    assert_size_stride(arg60_1, (384, 512), (512, 1))
    assert_size_stride(arg61_1, (512, 384), (384, 1))
    assert_size_stride(arg62_1, (1024, 512), (512, 1))
    assert_size_stride(arg63_1, (1024, 512), (512, 1))
    assert_size_stride(arg64_1, (512, 1024), (1024, 1))
    assert_size_stride(arg65_1, (384, 512), (512, 1))
    assert_size_stride(arg66_1, (384, 512), (512, 1))
    assert_size_stride(arg67_1, (384, 512), (512, 1))
    assert_size_stride(arg68_1, (512, 384), (384, 1))
    assert_size_stride(arg69_1, (1024, 512), (512, 1))
    assert_size_stride(arg70_1, (1024, 512), (512, 1))
    assert_size_stride(arg71_1, (512, 1024), (1024, 1))
    assert_size_stride(arg72_1, (384, 512), (512, 1))
    assert_size_stride(arg73_1, (384, 512), (512, 1))
    assert_size_stride(arg74_1, (384, 512), (512, 1))
    assert_size_stride(arg75_1, (512, 384), (384, 1))
    assert_size_stride(arg76_1, (1024, 512), (512, 1))
    assert_size_stride(arg77_1, (1024, 512), (512, 1))
    assert_size_stride(arg78_1, (512, 1024), (1024, 1))
    assert_size_stride(arg79_1, (384, 512), (512, 1))
    assert_size_stride(arg80_1, (384, 512), (512, 1))
    assert_size_stride(arg81_1, (384, 512), (512, 1))
    assert_size_stride(arg82_1, (512, 384), (384, 1))
    assert_size_stride(arg83_1, (1024, 512), (512, 1))
    assert_size_stride(arg84_1, (1024, 512), (512, 1))
    assert_size_stride(arg85_1, (512, 1024), (1024, 1))
    assert_size_stride(arg86_1, (384, 512), (512, 1))
    assert_size_stride(arg87_1, (384, 512), (512, 1))
    assert_size_stride(arg88_1, (384, 512), (512, 1))
    assert_size_stride(arg89_1, (512, 384), (384, 1))
    assert_size_stride(arg90_1, (1024, 512), (512, 1))
    assert_size_stride(arg91_1, (1024, 512), (512, 1))
    assert_size_stride(arg92_1, (512, 1024), (1024, 1))
    assert_size_stride(arg93_1, (384, 512), (512, 1))
    assert_size_stride(arg94_1, (384, 512), (512, 1))
    assert_size_stride(arg95_1, (384, 512), (512, 1))
    assert_size_stride(arg96_1, (512, 384), (384, 1))
    assert_size_stride(arg97_1, (1024, 512), (512, 1))
    assert_size_stride(arg98_1, (1024, 512), (512, 1))
    assert_size_stride(arg99_1, (512, 1024), (1024, 1))
    assert_size_stride(arg100_1, (384, 512), (512, 1))
    assert_size_stride(arg101_1, (384, 512), (512, 1))
    assert_size_stride(arg102_1, (384, 512), (512, 1))
    assert_size_stride(arg103_1, (32, 6), (6, 1))
    assert_size_stride(arg104_1, (512, 384), (384, 1))
    assert_size_stride(arg105_1, (384, 512), (512, 1))
    assert_size_stride(arg106_1, (384, 512), (512, 1))
    assert_size_stride(arg107_1, (384, 512), (512, 1))
    assert_size_stride(arg108_1, (512, 384), (384, 1))
    assert_size_stride(arg109_1, (1024, 512), (512, 1))
    assert_size_stride(arg110_1, (1024, 512), (512, 1))
    assert_size_stride(arg111_1, (512, 1024), (1024, 1))
    assert_size_stride(arg112_1, (384, 512), (512, 1))
    assert_size_stride(arg113_1, (384, 512), (512, 1))
    assert_size_stride(arg114_1, (384, 512), (512, 1))
    assert_size_stride(arg115_1, (512, 384), (384, 1))
    assert_size_stride(arg116_1, (384, 512), (512, 1))
    assert_size_stride(arg117_1, (384, 512), (512, 1))
    assert_size_stride(arg118_1, (384, 512), (512, 1))
    assert_size_stride(arg119_1, (512, 384), (384, 1))
    assert_size_stride(arg120_1, (1024, 512), (512, 1))
    assert_size_stride(arg121_1, (1024, 512), (512, 1))
    assert_size_stride(arg122_1, (512, 1024), (1024, 1))
    assert_size_stride(arg123_1, (384, 512), (512, 1))
    assert_size_stride(arg124_1, (384, 512), (512, 1))
    assert_size_stride(arg125_1, (384, 512), (512, 1))
    assert_size_stride(arg126_1, (512, 384), (384, 1))
    assert_size_stride(arg127_1, (384, 512), (512, 1))
    assert_size_stride(arg128_1, (384, 512), (512, 1))
    assert_size_stride(arg129_1, (384, 512), (512, 1))
    assert_size_stride(arg130_1, (512, 384), (384, 1))
    assert_size_stride(arg131_1, (1024, 512), (512, 1))
    assert_size_stride(arg132_1, (1024, 512), (512, 1))
    assert_size_stride(arg133_1, (512, 1024), (1024, 1))
    assert_size_stride(arg134_1, (384, 512), (512, 1))
    assert_size_stride(arg135_1, (384, 512), (512, 1))
    assert_size_stride(arg136_1, (384, 512), (512, 1))
    assert_size_stride(arg137_1, (512, 384), (384, 1))
    assert_size_stride(arg138_1, (384, 512), (512, 1))
    assert_size_stride(arg139_1, (384, 512), (512, 1))
    assert_size_stride(arg140_1, (384, 512), (512, 1))
    assert_size_stride(arg141_1, (512, 384), (384, 1))
    assert_size_stride(arg142_1, (1024, 512), (512, 1))
    assert_size_stride(arg143_1, (1024, 512), (512, 1))
    assert_size_stride(arg144_1, (512, 1024), (1024, 1))
    assert_size_stride(arg145_1, (384, 512), (512, 1))
    assert_size_stride(arg146_1, (384, 512), (512, 1))
    assert_size_stride(arg147_1, (384, 512), (512, 1))
    assert_size_stride(arg148_1, (512, 384), (384, 1))
    assert_size_stride(arg149_1, (384, 512), (512, 1))
    assert_size_stride(arg150_1, (384, 512), (512, 1))
    assert_size_stride(arg151_1, (384, 512), (512, 1))
    assert_size_stride(arg152_1, (512, 384), (384, 1))
    assert_size_stride(arg153_1, (1024, 512), (512, 1))
    assert_size_stride(arg154_1, (1024, 512), (512, 1))
    assert_size_stride(arg155_1, (512, 1024), (1024, 1))
    assert_size_stride(arg156_1, (384, 512), (512, 1))
    assert_size_stride(arg157_1, (384, 512), (512, 1))
    assert_size_stride(arg158_1, (384, 512), (512, 1))
    assert_size_stride(arg159_1, (512, 384), (384, 1))
    assert_size_stride(arg160_1, (384, 512), (512, 1))
    assert_size_stride(arg161_1, (384, 512), (512, 1))
    assert_size_stride(arg162_1, (384, 512), (512, 1))
    assert_size_stride(arg163_1, (512, 384), (384, 1))
    assert_size_stride(arg164_1, (1024, 512), (512, 1))
    assert_size_stride(arg165_1, (1024, 512), (512, 1))
    assert_size_stride(arg166_1, (512, 1024), (1024, 1))
    assert_size_stride(arg167_1, (384, 512), (512, 1))
    assert_size_stride(arg168_1, (384, 512), (512, 1))
    assert_size_stride(arg169_1, (384, 512), (512, 1))
    assert_size_stride(arg170_1, (512, 384), (384, 1))
    assert_size_stride(arg171_1, (384, 512), (512, 1))
    assert_size_stride(arg172_1, (384, 512), (512, 1))
    assert_size_stride(arg173_1, (384, 512), (512, 1))
    assert_size_stride(arg174_1, (512, 384), (384, 1))
    assert_size_stride(arg175_1, (1024, 512), (512, 1))
    assert_size_stride(arg176_1, (1024, 512), (512, 1))
    assert_size_stride(arg177_1, (512, 1024), (1024, 1))
    assert_size_stride(arg178_1, (384, 512), (512, 1))
    assert_size_stride(arg179_1, (384, 512), (512, 1))
    assert_size_stride(arg180_1, (384, 512), (512, 1))
    assert_size_stride(arg181_1, (512, 384), (384, 1))
    assert_size_stride(arg182_1, (384, 512), (512, 1))
    assert_size_stride(arg183_1, (384, 512), (512, 1))
    assert_size_stride(arg184_1, (384, 512), (512, 1))
    assert_size_stride(arg185_1, (512, 384), (384, 1))
    assert_size_stride(arg186_1, (1024, 512), (512, 1))
    assert_size_stride(arg187_1, (1024, 512), (512, 1))
    assert_size_stride(arg188_1, (512, 1024), (1024, 1))
    assert_size_stride(arg189_1, (250112, 512), (512, 1))
    assert_size_stride(arg190_1, (1, 128), (128, 1))
    assert_size_stride(arg191_1, (1, 128), (128, 1))
    assert_size_stride(arg192_1, (1, 128), (128, 1))
    buf0 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf1 = empty((1, 128, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_embedding_mean_mul_pow_rsqrt_0(c_void_p(arg192_1.data_ptr()), c_void_p(arg42_1.data_ptr()), c_void_p(arg17_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg17_1
    buf2 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_0_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf1, (128, 512), (512, 1), 0), reinterpret_tensor(arg100_1, (512, 384), (1, 512), 0), out=buf2)
    del arg100_1
    buf3 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___decoder_block_0_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf1, (128, 512), (512, 1), 0), reinterpret_tensor(arg101_1, (512, 384), (1, 512), 0), out=buf3)
    del arg101_1
    buf4 = empty((6, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [scores_16], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf2, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf3, (6, 64, 128), (64, 1, 384), 0), out=buf4)
    buf5 = empty_strided((1, 6, 128, 1), (768, 128, 1, 768), device='cpu', dtype=torch.float32)
    buf6 = reinterpret_tensor(buf4, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf4  # reuse
    buf7 = buf6; del buf6  # reuse
    buf8 = empty_strided((1, 6, 128, 1), (768, 128, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_1(c_void_p(buf7.data_ptr()), c_void_p(arg103_1.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf8.data_ptr()))
    buf9 = buf2; del buf2  # reuse
    # Source Nodes: [l__mod___decoder_block_0_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf1, (128, 512), (512, 1), 0), reinterpret_tensor(arg102_1, (512, 384), (1, 512), 0), out=buf9)
    del arg102_1
    buf10 = buf7; del buf7  # reuse
    cpp_fused__softmax_2(c_void_p(buf10.data_ptr()), c_void_p(buf8.data_ptr()))
    buf11 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_17], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf10, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf9, (6, 128, 64), (64, 384, 1), 0), out=buf11)
    buf12 = empty((1, 128, 6, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_3(c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()))
    buf13 = reinterpret_tensor(buf1, (128, 512), (512, 1), 0); del buf1  # reuse
    # Source Nodes: [attn_output_17], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf12, (128, 384), (384, 1), 0), reinterpret_tensor(arg104_1, (384, 512), (1, 384), 0), out=buf13)
    del arg104_1
    buf14 = buf0; del buf0  # reuse
    buf17 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf15 = empty((1, 128, 512), device='cpu', dtype=torch.float32)
    buf18 = empty((1, 128, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_embedding_mean_mul_pow_rsqrt_4(c_void_p(arg192_1.data_ptr()), c_void_p(arg42_1.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(arg190_1.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf18.data_ptr()))
    del arg0_1
    del arg18_1
    buf16 = reinterpret_tensor(buf12, (128, 384), (384, 1), 0); del buf12  # reuse
    # Source Nodes: [l__mod___decoder_block_0_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf15, (128, 512), (512, 1), 0), reinterpret_tensor(arg105_1, (512, 384), (1, 512), 0), out=buf16)
    del arg105_1
    buf19 = reinterpret_tensor(buf11, (128, 384), (384, 1), 0); del buf11  # reuse
    # Source Nodes: [l__mod___encoder_block_0_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf18, (128, 512), (512, 1), 0), reinterpret_tensor(arg43_1, (512, 384), (1, 512), 0), out=buf19)
    del arg43_1
    buf20 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_0_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf18, (128, 512), (512, 1), 0), reinterpret_tensor(arg44_1, (512, 384), (1, 512), 0), out=buf20)
    del arg44_1
    buf21 = reinterpret_tensor(buf10, (6, 128, 128), (16384, 128, 1), 0); del buf10  # reuse
    # Source Nodes: [scores], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf19, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf20, (6, 64, 128), (64, 1, 384), 0), out=buf21)
    buf22 = buf8; del buf8  # reuse
    buf23 = reinterpret_tensor(buf21, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf21  # reuse
    buf24 = buf5; del buf5  # reuse
    cpp_fused__softmax_5(c_void_p(buf23.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf24.data_ptr()))
    buf25 = buf20; del buf20  # reuse
    # Source Nodes: [l__mod___encoder_block_0_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf18, (128, 512), (512, 1), 0), reinterpret_tensor(arg45_1, (512, 384), (1, 512), 0), out=buf25)
    del arg45_1
    buf26 = buf23; del buf23  # reuse
    cpp_fused__softmax_6(c_void_p(buf26.data_ptr()), c_void_p(buf24.data_ptr()))
    buf27 = reinterpret_tensor(buf19, (6, 128, 64), (8192, 64, 1), 0); del buf19  # reuse
    # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf26, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf25, (6, 128, 64), (64, 384, 1), 0), out=buf27)
    buf28 = reinterpret_tensor(buf25, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf25  # reuse
    cpp_fused_clone_7(c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()))
    buf29 = reinterpret_tensor(buf18, (128, 512), (512, 1), 0); del buf18  # reuse
    # Source Nodes: [attn_output_1], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf28, (128, 384), (384, 1), 0), reinterpret_tensor(arg47_1, (384, 512), (1, 384), 0), out=buf29)
    del arg47_1
    buf30 = buf17; del buf17  # reuse
    buf31 = buf15; del buf15  # reuse
    cpp_fused_add_embedding_mean_mul_pow_rsqrt_8(c_void_p(arg190_1.data_ptr()), c_void_p(arg42_1.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()))
    del arg1_1
    buf32 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___encoder_block_0_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf31, (128, 512), (512, 1), 0), reinterpret_tensor(arg48_1, (512, 1024), (1, 512), 0), out=buf32)
    del arg48_1
    buf33 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_linear], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf31, (128, 512), (512, 1), 0), reinterpret_tensor(arg49_1, (512, 1024), (1, 512), 0), out=buf33)
    del arg49_1
    buf34 = reinterpret_tensor(buf32, (1, 128, 1024), (131072, 1024, 1), 0); del buf32  # reuse
    cpp_fused_add_mul_pow_tanh_9(c_void_p(buf34.data_ptr()), c_void_p(buf33.data_ptr()))
    buf35 = reinterpret_tensor(buf31, (128, 512), (512, 1), 0); del buf31  # reuse
    # Source Nodes: [forwarded_states_1], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf34, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg50_1, (1024, 512), (1, 1024), 0), out=buf35)
    del arg50_1
    buf36 = buf30; del buf30  # reuse
    buf37 = empty((1, 128, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_embedding_mean_mul_pow_rsqrt_10(c_void_p(arg190_1.data_ptr()), c_void_p(arg42_1.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()))
    del arg2_1
    buf38 = reinterpret_tensor(buf28, (128, 384), (384, 1), 0); del buf28  # reuse
    # Source Nodes: [l__mod___encoder_block_1_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf37, (128, 512), (512, 1), 0), reinterpret_tensor(arg51_1, (512, 384), (1, 512), 0), out=buf38)
    del arg51_1
    buf39 = reinterpret_tensor(buf27, (128, 384), (384, 1), 0); del buf27  # reuse
    # Source Nodes: [l__mod___encoder_block_1_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf37, (128, 512), (512, 1), 0), reinterpret_tensor(arg52_1, (512, 384), (1, 512), 0), out=buf39)
    del arg52_1
    buf40 = reinterpret_tensor(buf26, (6, 128, 128), (16384, 128, 1), 0); del buf26  # reuse
    # Source Nodes: [scores_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf38, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf39, (6, 64, 128), (64, 1, 384), 0), out=buf40)
    buf41 = buf24; del buf24  # reuse
    buf42 = reinterpret_tensor(buf40, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf40  # reuse
    buf43 = buf22; del buf22  # reuse
    cpp_fused__softmax_11(c_void_p(buf42.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf43.data_ptr()))
    buf44 = buf39; del buf39  # reuse
    # Source Nodes: [l__mod___encoder_block_1_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf37, (128, 512), (512, 1), 0), reinterpret_tensor(arg53_1, (512, 384), (1, 512), 0), out=buf44)
    del arg53_1
    buf45 = buf42; del buf42  # reuse
    cpp_fused__softmax_12(c_void_p(buf45.data_ptr()), c_void_p(buf43.data_ptr()))
    buf46 = reinterpret_tensor(buf38, (6, 128, 64), (8192, 64, 1), 0); del buf38  # reuse
    # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf45, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf44, (6, 128, 64), (64, 384, 1), 0), out=buf46)
    buf47 = reinterpret_tensor(buf44, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf44  # reuse
    cpp_fused_clone_13(c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()))
    buf48 = reinterpret_tensor(buf37, (128, 512), (512, 1), 0); del buf37  # reuse
    # Source Nodes: [attn_output_3], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf47, (128, 384), (384, 1), 0), reinterpret_tensor(arg54_1, (384, 512), (1, 384), 0), out=buf48)
    del arg54_1
    buf49 = reinterpret_tensor(buf29, (1, 128, 512), (65536, 512, 1), 0); del buf29  # reuse
    buf50 = buf36; del buf36  # reuse
    buf51 = empty((1, 128, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_embedding_mean_mul_pow_rsqrt_14(c_void_p(buf49.data_ptr()), c_void_p(arg190_1.data_ptr()), c_void_p(arg42_1.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()))
    del arg190_1
    del arg3_1
    buf52 = reinterpret_tensor(buf34, (128, 1024), (1024, 1), 0); del buf34  # reuse
    # Source Nodes: [l__mod___encoder_block_1_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf51, (128, 512), (512, 1), 0), reinterpret_tensor(arg55_1, (512, 1024), (1, 512), 0), out=buf52)
    del arg55_1
    buf53 = buf33; del buf33  # reuse
    # Source Nodes: [hidden_linear_1], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf51, (128, 512), (512, 1), 0), reinterpret_tensor(arg56_1, (512, 1024), (1, 512), 0), out=buf53)
    del arg56_1
    buf54 = reinterpret_tensor(buf52, (1, 128, 1024), (131072, 1024, 1), 0); del buf52  # reuse
    cpp_fused_add_mul_pow_tanh_15(c_void_p(buf54.data_ptr()), c_void_p(buf53.data_ptr()))
    buf55 = reinterpret_tensor(buf51, (128, 512), (512, 1), 0); del buf51  # reuse
    # Source Nodes: [forwarded_states_3], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf54, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg57_1, (1024, 512), (1, 1024), 0), out=buf55)
    del arg57_1
    buf56 = buf50; del buf50  # reuse
    buf57 = reinterpret_tensor(buf48, (1, 128, 512), (65536, 512, 1), 0); del buf48  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_16(c_void_p(buf49.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf57.data_ptr()))
    del arg4_1
    buf58 = reinterpret_tensor(buf47, (128, 384), (384, 1), 0); del buf47  # reuse
    # Source Nodes: [l__mod___encoder_block_2_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf57, (128, 512), (512, 1), 0), reinterpret_tensor(arg58_1, (512, 384), (1, 512), 0), out=buf58)
    del arg58_1
    buf59 = reinterpret_tensor(buf46, (128, 384), (384, 1), 0); del buf46  # reuse
    # Source Nodes: [l__mod___encoder_block_2_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf57, (128, 512), (512, 1), 0), reinterpret_tensor(arg59_1, (512, 384), (1, 512), 0), out=buf59)
    del arg59_1
    buf60 = reinterpret_tensor(buf45, (6, 128, 128), (16384, 128, 1), 0); del buf45  # reuse
    # Source Nodes: [scores_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf58, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf59, (6, 64, 128), (64, 1, 384), 0), out=buf60)
    buf61 = buf43; del buf43  # reuse
    buf62 = reinterpret_tensor(buf60, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf60  # reuse
    buf63 = buf41; del buf41  # reuse
    cpp_fused__softmax_17(c_void_p(buf62.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf63.data_ptr()))
    buf64 = buf59; del buf59  # reuse
    # Source Nodes: [l__mod___encoder_block_2_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf57, (128, 512), (512, 1), 0), reinterpret_tensor(arg60_1, (512, 384), (1, 512), 0), out=buf64)
    del arg60_1
    buf65 = buf62; del buf62  # reuse
    cpp_fused__softmax_18(c_void_p(buf65.data_ptr()), c_void_p(buf63.data_ptr()))
    buf66 = reinterpret_tensor(buf58, (6, 128, 64), (8192, 64, 1), 0); del buf58  # reuse
    # Source Nodes: [matmul_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf65, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf64, (6, 128, 64), (64, 384, 1), 0), out=buf66)
    buf67 = reinterpret_tensor(buf64, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf64  # reuse
    cpp_fused_clone_19(c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()))
    buf68 = reinterpret_tensor(buf57, (128, 512), (512, 1), 0); del buf57  # reuse
    # Source Nodes: [attn_output_5], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf67, (128, 384), (384, 1), 0), reinterpret_tensor(arg61_1, (384, 512), (1, 384), 0), out=buf68)
    del arg61_1
    buf69 = buf56; del buf56  # reuse
    buf70 = reinterpret_tensor(buf35, (1, 128, 512), (65536, 512, 1), 0); del buf35  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_20(c_void_p(buf49.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(arg5_1.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()))
    del arg5_1
    buf71 = reinterpret_tensor(buf54, (128, 1024), (1024, 1), 0); del buf54  # reuse
    # Source Nodes: [l__mod___encoder_block_2_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf70, (128, 512), (512, 1), 0), reinterpret_tensor(arg62_1, (512, 1024), (1, 512), 0), out=buf71)
    del arg62_1
    buf72 = buf53; del buf53  # reuse
    # Source Nodes: [hidden_linear_2], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf70, (128, 512), (512, 1), 0), reinterpret_tensor(arg63_1, (512, 1024), (1, 512), 0), out=buf72)
    del arg63_1
    buf73 = reinterpret_tensor(buf71, (1, 128, 1024), (131072, 1024, 1), 0); del buf71  # reuse
    cpp_fused_add_mul_pow_tanh_21(c_void_p(buf73.data_ptr()), c_void_p(buf72.data_ptr()))
    buf74 = reinterpret_tensor(buf70, (128, 512), (512, 1), 0); del buf70  # reuse
    # Source Nodes: [forwarded_states_5], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf73, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg64_1, (1024, 512), (1, 1024), 0), out=buf74)
    del arg64_1
    buf75 = buf69; del buf69  # reuse
    buf76 = empty((1, 128, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_22(c_void_p(buf49.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()))
    del arg6_1
    buf77 = reinterpret_tensor(buf67, (128, 384), (384, 1), 0); del buf67  # reuse
    # Source Nodes: [l__mod___encoder_block_3_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf76, (128, 512), (512, 1), 0), reinterpret_tensor(arg65_1, (512, 384), (1, 512), 0), out=buf77)
    del arg65_1
    buf78 = reinterpret_tensor(buf66, (128, 384), (384, 1), 0); del buf66  # reuse
    # Source Nodes: [l__mod___encoder_block_3_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf76, (128, 512), (512, 1), 0), reinterpret_tensor(arg66_1, (512, 384), (1, 512), 0), out=buf78)
    del arg66_1
    buf79 = reinterpret_tensor(buf65, (6, 128, 128), (16384, 128, 1), 0); del buf65  # reuse
    # Source Nodes: [scores_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf77, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf78, (6, 64, 128), (64, 1, 384), 0), out=buf79)
    buf80 = buf63; del buf63  # reuse
    buf81 = reinterpret_tensor(buf79, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf79  # reuse
    buf82 = buf61; del buf61  # reuse
    cpp_fused__softmax_23(c_void_p(buf81.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf82.data_ptr()))
    buf83 = buf78; del buf78  # reuse
    # Source Nodes: [l__mod___encoder_block_3_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf76, (128, 512), (512, 1), 0), reinterpret_tensor(arg67_1, (512, 384), (1, 512), 0), out=buf83)
    del arg67_1
    buf84 = buf81; del buf81  # reuse
    cpp_fused__softmax_24(c_void_p(buf84.data_ptr()), c_void_p(buf82.data_ptr()))
    buf85 = reinterpret_tensor(buf77, (6, 128, 64), (8192, 64, 1), 0); del buf77  # reuse
    # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf84, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf83, (6, 128, 64), (64, 384, 1), 0), out=buf85)
    buf86 = reinterpret_tensor(buf83, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf83  # reuse
    cpp_fused_clone_25(c_void_p(buf85.data_ptr()), c_void_p(buf86.data_ptr()))
    buf87 = reinterpret_tensor(buf76, (128, 512), (512, 1), 0); del buf76  # reuse
    # Source Nodes: [attn_output_7], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf86, (128, 384), (384, 1), 0), reinterpret_tensor(arg68_1, (384, 512), (1, 384), 0), out=buf87)
    del arg68_1
    buf88 = buf49; del buf49  # reuse
    buf89 = buf75; del buf75  # reuse
    buf90 = empty((1, 128, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mean_mul_pow_rsqrt_26(c_void_p(buf88.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(arg7_1.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf90.data_ptr()))
    del arg7_1
    buf91 = reinterpret_tensor(buf73, (128, 1024), (1024, 1), 0); del buf73  # reuse
    # Source Nodes: [l__mod___encoder_block_3_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf90, (128, 512), (512, 1), 0), reinterpret_tensor(arg69_1, (512, 1024), (1, 512), 0), out=buf91)
    del arg69_1
    buf92 = buf72; del buf72  # reuse
    # Source Nodes: [hidden_linear_3], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf90, (128, 512), (512, 1), 0), reinterpret_tensor(arg70_1, (512, 1024), (1, 512), 0), out=buf92)
    del arg70_1
    buf93 = reinterpret_tensor(buf91, (1, 128, 1024), (131072, 1024, 1), 0); del buf91  # reuse
    cpp_fused_add_mul_pow_tanh_27(c_void_p(buf93.data_ptr()), c_void_p(buf92.data_ptr()))
    buf94 = reinterpret_tensor(buf90, (128, 512), (512, 1), 0); del buf90  # reuse
    # Source Nodes: [forwarded_states_7], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf93, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg71_1, (1024, 512), (1, 1024), 0), out=buf94)
    del arg71_1
    buf95 = buf89; del buf89  # reuse
    buf96 = reinterpret_tensor(buf87, (1, 128, 512), (65536, 512, 1), 0); del buf87  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_28(c_void_p(buf88.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf96.data_ptr()))
    del arg8_1
    buf97 = reinterpret_tensor(buf86, (128, 384), (384, 1), 0); del buf86  # reuse
    # Source Nodes: [l__mod___encoder_block_4_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf96, (128, 512), (512, 1), 0), reinterpret_tensor(arg72_1, (512, 384), (1, 512), 0), out=buf97)
    del arg72_1
    buf98 = reinterpret_tensor(buf85, (128, 384), (384, 1), 0); del buf85  # reuse
    # Source Nodes: [l__mod___encoder_block_4_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf96, (128, 512), (512, 1), 0), reinterpret_tensor(arg73_1, (512, 384), (1, 512), 0), out=buf98)
    del arg73_1
    buf99 = reinterpret_tensor(buf84, (6, 128, 128), (16384, 128, 1), 0); del buf84  # reuse
    # Source Nodes: [scores_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf97, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf98, (6, 64, 128), (64, 1, 384), 0), out=buf99)
    buf100 = buf82; del buf82  # reuse
    buf101 = reinterpret_tensor(buf99, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf99  # reuse
    buf102 = buf80; del buf80  # reuse
    cpp_fused__softmax_29(c_void_p(buf101.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf102.data_ptr()))
    buf103 = buf98; del buf98  # reuse
    # Source Nodes: [l__mod___encoder_block_4_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf96, (128, 512), (512, 1), 0), reinterpret_tensor(arg74_1, (512, 384), (1, 512), 0), out=buf103)
    del arg74_1
    buf104 = buf101; del buf101  # reuse
    cpp_fused__softmax_30(c_void_p(buf104.data_ptr()), c_void_p(buf102.data_ptr()))
    buf105 = reinterpret_tensor(buf97, (6, 128, 64), (8192, 64, 1), 0); del buf97  # reuse
    # Source Nodes: [matmul_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf104, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf103, (6, 128, 64), (64, 384, 1), 0), out=buf105)
    buf106 = reinterpret_tensor(buf103, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf103  # reuse
    cpp_fused_clone_31(c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()))
    buf107 = reinterpret_tensor(buf96, (128, 512), (512, 1), 0); del buf96  # reuse
    # Source Nodes: [attn_output_9], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf106, (128, 384), (384, 1), 0), reinterpret_tensor(arg75_1, (384, 512), (1, 384), 0), out=buf107)
    del arg75_1
    buf108 = buf95; del buf95  # reuse
    buf109 = reinterpret_tensor(buf74, (1, 128, 512), (65536, 512, 1), 0); del buf74  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_32(c_void_p(buf88.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(arg9_1.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()))
    del arg9_1
    buf110 = reinterpret_tensor(buf93, (128, 1024), (1024, 1), 0); del buf93  # reuse
    # Source Nodes: [l__mod___encoder_block_4_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf109, (128, 512), (512, 1), 0), reinterpret_tensor(arg76_1, (512, 1024), (1, 512), 0), out=buf110)
    del arg76_1
    buf111 = buf92; del buf92  # reuse
    # Source Nodes: [hidden_linear_4], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf109, (128, 512), (512, 1), 0), reinterpret_tensor(arg77_1, (512, 1024), (1, 512), 0), out=buf111)
    del arg77_1
    buf112 = reinterpret_tensor(buf110, (1, 128, 1024), (131072, 1024, 1), 0); del buf110  # reuse
    cpp_fused_add_mul_pow_tanh_33(c_void_p(buf112.data_ptr()), c_void_p(buf111.data_ptr()))
    buf113 = reinterpret_tensor(buf109, (128, 512), (512, 1), 0); del buf109  # reuse
    # Source Nodes: [forwarded_states_9], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf112, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg78_1, (1024, 512), (1, 1024), 0), out=buf113)
    del arg78_1
    buf114 = buf108; del buf108  # reuse
    buf115 = reinterpret_tensor(buf68, (1, 128, 512), (65536, 512, 1), 0); del buf68  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_34(c_void_p(buf88.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf115.data_ptr()))
    del arg10_1
    buf116 = reinterpret_tensor(buf106, (128, 384), (384, 1), 0); del buf106  # reuse
    # Source Nodes: [l__mod___encoder_block_5_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf115, (128, 512), (512, 1), 0), reinterpret_tensor(arg79_1, (512, 384), (1, 512), 0), out=buf116)
    del arg79_1
    buf117 = reinterpret_tensor(buf105, (128, 384), (384, 1), 0); del buf105  # reuse
    # Source Nodes: [l__mod___encoder_block_5_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf115, (128, 512), (512, 1), 0), reinterpret_tensor(arg80_1, (512, 384), (1, 512), 0), out=buf117)
    del arg80_1
    buf118 = reinterpret_tensor(buf104, (6, 128, 128), (16384, 128, 1), 0); del buf104  # reuse
    # Source Nodes: [scores_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf116, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf117, (6, 64, 128), (64, 1, 384), 0), out=buf118)
    buf119 = buf102; del buf102  # reuse
    buf120 = reinterpret_tensor(buf118, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf118  # reuse
    buf121 = buf100; del buf100  # reuse
    cpp_fused__softmax_35(c_void_p(buf120.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf121.data_ptr()))
    buf122 = buf117; del buf117  # reuse
    # Source Nodes: [l__mod___encoder_block_5_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf115, (128, 512), (512, 1), 0), reinterpret_tensor(arg81_1, (512, 384), (1, 512), 0), out=buf122)
    del arg81_1
    buf123 = buf120; del buf120  # reuse
    cpp_fused__softmax_36(c_void_p(buf123.data_ptr()), c_void_p(buf121.data_ptr()))
    buf124 = reinterpret_tensor(buf116, (6, 128, 64), (8192, 64, 1), 0); del buf116  # reuse
    # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf123, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf122, (6, 128, 64), (64, 384, 1), 0), out=buf124)
    buf125 = reinterpret_tensor(buf122, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf122  # reuse
    cpp_fused_clone_37(c_void_p(buf124.data_ptr()), c_void_p(buf125.data_ptr()))
    buf126 = reinterpret_tensor(buf115, (128, 512), (512, 1), 0); del buf115  # reuse
    # Source Nodes: [attn_output_11], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf125, (128, 384), (384, 1), 0), reinterpret_tensor(arg82_1, (384, 512), (1, 384), 0), out=buf126)
    del arg82_1
    buf127 = reinterpret_tensor(buf107, (1, 128, 512), (65536, 512, 1), 0); del buf107  # reuse
    buf128 = buf114; del buf114  # reuse
    buf129 = reinterpret_tensor(buf55, (1, 128, 512), (65536, 512, 1), 0); del buf55  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_38(c_void_p(buf127.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(arg11_1.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf129.data_ptr()))
    del arg11_1
    buf130 = reinterpret_tensor(buf112, (128, 1024), (1024, 1), 0); del buf112  # reuse
    # Source Nodes: [l__mod___encoder_block_5_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf129, (128, 512), (512, 1), 0), reinterpret_tensor(arg83_1, (512, 1024), (1, 512), 0), out=buf130)
    del arg83_1
    buf131 = buf111; del buf111  # reuse
    # Source Nodes: [hidden_linear_5], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf129, (128, 512), (512, 1), 0), reinterpret_tensor(arg84_1, (512, 1024), (1, 512), 0), out=buf131)
    del arg84_1
    buf132 = reinterpret_tensor(buf130, (1, 128, 1024), (131072, 1024, 1), 0); del buf130  # reuse
    cpp_fused_add_mul_pow_tanh_39(c_void_p(buf132.data_ptr()), c_void_p(buf131.data_ptr()))
    buf133 = reinterpret_tensor(buf129, (128, 512), (512, 1), 0); del buf129  # reuse
    # Source Nodes: [forwarded_states_11], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf132, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg85_1, (1024, 512), (1, 1024), 0), out=buf133)
    del arg85_1
    buf134 = buf128; del buf128  # reuse
    buf135 = reinterpret_tensor(buf94, (1, 128, 512), (65536, 512, 1), 0); del buf94  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_40(c_void_p(buf127.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf135.data_ptr()))
    del arg12_1
    buf136 = reinterpret_tensor(buf125, (128, 384), (384, 1), 0); del buf125  # reuse
    # Source Nodes: [l__mod___encoder_block_6_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf135, (128, 512), (512, 1), 0), reinterpret_tensor(arg86_1, (512, 384), (1, 512), 0), out=buf136)
    del arg86_1
    buf137 = reinterpret_tensor(buf124, (128, 384), (384, 1), 0); del buf124  # reuse
    # Source Nodes: [l__mod___encoder_block_6_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf135, (128, 512), (512, 1), 0), reinterpret_tensor(arg87_1, (512, 384), (1, 512), 0), out=buf137)
    del arg87_1
    buf138 = reinterpret_tensor(buf123, (6, 128, 128), (16384, 128, 1), 0); del buf123  # reuse
    # Source Nodes: [scores_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf136, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf137, (6, 64, 128), (64, 1, 384), 0), out=buf138)
    buf139 = buf121; del buf121  # reuse
    buf140 = reinterpret_tensor(buf138, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf138  # reuse
    buf141 = buf119; del buf119  # reuse
    cpp_fused__softmax_41(c_void_p(buf140.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf141.data_ptr()))
    buf142 = buf137; del buf137  # reuse
    # Source Nodes: [l__mod___encoder_block_6_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf135, (128, 512), (512, 1), 0), reinterpret_tensor(arg88_1, (512, 384), (1, 512), 0), out=buf142)
    del arg88_1
    buf143 = buf140; del buf140  # reuse
    cpp_fused__softmax_42(c_void_p(buf143.data_ptr()), c_void_p(buf141.data_ptr()))
    buf144 = reinterpret_tensor(buf136, (6, 128, 64), (8192, 64, 1), 0); del buf136  # reuse
    # Source Nodes: [matmul_13], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf143, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf142, (6, 128, 64), (64, 384, 1), 0), out=buf144)
    buf145 = reinterpret_tensor(buf142, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf142  # reuse
    cpp_fused_clone_43(c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()))
    buf146 = reinterpret_tensor(buf135, (128, 512), (512, 1), 0); del buf135  # reuse
    # Source Nodes: [attn_output_13], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf145, (128, 384), (384, 1), 0), reinterpret_tensor(arg89_1, (384, 512), (1, 384), 0), out=buf146)
    del arg89_1
    buf147 = buf134; del buf134  # reuse
    buf148 = buf88; del buf88  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_44(c_void_p(buf127.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf148.data_ptr()))
    del arg13_1
    buf149 = reinterpret_tensor(buf132, (128, 1024), (1024, 1), 0); del buf132  # reuse
    # Source Nodes: [l__mod___encoder_block_6_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf148, (128, 512), (512, 1), 0), reinterpret_tensor(arg90_1, (512, 1024), (1, 512), 0), out=buf149)
    del arg90_1
    buf150 = buf131; del buf131  # reuse
    # Source Nodes: [hidden_linear_6], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf148, (128, 512), (512, 1), 0), reinterpret_tensor(arg91_1, (512, 1024), (1, 512), 0), out=buf150)
    del arg91_1
    buf151 = reinterpret_tensor(buf149, (1, 128, 1024), (131072, 1024, 1), 0); del buf149  # reuse
    cpp_fused_add_mul_pow_tanh_45(c_void_p(buf151.data_ptr()), c_void_p(buf150.data_ptr()))
    buf152 = reinterpret_tensor(buf148, (128, 512), (512, 1), 0); del buf148  # reuse
    # Source Nodes: [forwarded_states_13], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf151, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg92_1, (1024, 512), (1, 1024), 0), out=buf152)
    del arg92_1
    buf153 = buf147; del buf147  # reuse
    buf154 = reinterpret_tensor(buf126, (1, 128, 512), (65536, 512, 1), 0); del buf126  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_46(c_void_p(buf127.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()))
    del arg14_1
    buf155 = reinterpret_tensor(buf145, (128, 384), (384, 1), 0); del buf145  # reuse
    # Source Nodes: [l__mod___encoder_block_7_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf154, (128, 512), (512, 1), 0), reinterpret_tensor(arg93_1, (512, 384), (1, 512), 0), out=buf155)
    del arg93_1
    buf156 = reinterpret_tensor(buf144, (128, 384), (384, 1), 0); del buf144  # reuse
    # Source Nodes: [l__mod___encoder_block_7_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf154, (128, 512), (512, 1), 0), reinterpret_tensor(arg94_1, (512, 384), (1, 512), 0), out=buf156)
    del arg94_1
    buf157 = reinterpret_tensor(buf143, (6, 128, 128), (16384, 128, 1), 0); del buf143  # reuse
    # Source Nodes: [scores_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf155, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf156, (6, 64, 128), (64, 1, 384), 0), out=buf157)
    buf158 = buf141; del buf141  # reuse
    buf159 = reinterpret_tensor(buf157, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf157  # reuse
    buf160 = buf139; del buf139  # reuse
    cpp_fused__softmax_47(c_void_p(buf159.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf160.data_ptr()))
    del arg46_1
    buf161 = buf156; del buf156  # reuse
    # Source Nodes: [l__mod___encoder_block_7_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf154, (128, 512), (512, 1), 0), reinterpret_tensor(arg95_1, (512, 384), (1, 512), 0), out=buf161)
    del arg95_1
    buf162 = buf159; del buf159  # reuse
    cpp_fused__softmax_48(c_void_p(buf162.data_ptr()), c_void_p(buf160.data_ptr()))
    buf163 = reinterpret_tensor(buf155, (6, 128, 64), (8192, 64, 1), 0); del buf155  # reuse
    # Source Nodes: [matmul_15], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf162, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf161, (6, 128, 64), (64, 384, 1), 0), out=buf163)
    buf164 = reinterpret_tensor(buf161, (1, 128, 6, 64), (49152, 384, 64, 1), 0); del buf161  # reuse
    cpp_fused_clone_49(c_void_p(buf163.data_ptr()), c_void_p(buf164.data_ptr()))
    buf165 = reinterpret_tensor(buf154, (128, 512), (512, 1), 0); del buf154  # reuse
    # Source Nodes: [attn_output_15], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf164, (128, 384), (384, 1), 0), reinterpret_tensor(arg96_1, (384, 512), (1, 384), 0), out=buf165)
    del arg96_1
    buf166 = buf127; del buf127  # reuse
    buf167 = buf153; del buf153  # reuse
    buf168 = reinterpret_tensor(buf113, (1, 128, 512), (65536, 512, 1), 0); del buf113  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_50(c_void_p(buf166.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf168.data_ptr()))
    del arg15_1
    buf169 = reinterpret_tensor(buf151, (128, 1024), (1024, 1), 0); del buf151  # reuse
    # Source Nodes: [l__mod___encoder_block_7_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf168, (128, 512), (512, 1), 0), reinterpret_tensor(arg97_1, (512, 1024), (1, 512), 0), out=buf169)
    del arg97_1
    buf170 = buf150; del buf150  # reuse
    # Source Nodes: [hidden_linear_7], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf168, (128, 512), (512, 1), 0), reinterpret_tensor(arg98_1, (512, 1024), (1, 512), 0), out=buf170)
    del arg98_1
    buf171 = reinterpret_tensor(buf169, (1, 128, 1024), (131072, 1024, 1), 0); del buf169  # reuse
    cpp_fused_add_mul_pow_tanh_51(c_void_p(buf171.data_ptr()), c_void_p(buf170.data_ptr()))
    buf172 = reinterpret_tensor(buf168, (128, 512), (512, 1), 0); del buf168  # reuse
    # Source Nodes: [forwarded_states_15], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf171, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg99_1, (1024, 512), (1, 1024), 0), out=buf172)
    del arg99_1
    buf173 = buf167; del buf167  # reuse
    buf174 = buf166; del buf166  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_52(c_void_p(buf174.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(buf173.data_ptr()))
    del arg16_1
    buf175 = reinterpret_tensor(buf164, (128, 384), (384, 1), 0); del buf164  # reuse
    # Source Nodes: [l__mod___decoder_block_0_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf174, (128, 512), (512, 1), 0), reinterpret_tensor(arg106_1, (512, 384), (1, 512), 0), out=buf175)
    del arg106_1
    buf176 = reinterpret_tensor(buf162, (6, 128, 128), (16384, 128, 1), 0); del buf162  # reuse
    # Source Nodes: [scores_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf16, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf175, (6, 64, 128), (64, 1, 384), 0), out=buf176)
    buf177 = buf160; del buf160  # reuse
    buf178 = reinterpret_tensor(buf176, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf176  # reuse
    buf179 = buf158; del buf158  # reuse
    cpp_fused__softmax_53(c_void_p(buf178.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf179.data_ptr()))
    buf180 = buf16; del buf16  # reuse
    # Source Nodes: [l__mod___decoder_block_0_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf174, (128, 512), (512, 1), 0), reinterpret_tensor(arg107_1, (512, 384), (1, 512), 0), out=buf180)
    del arg107_1
    buf181 = buf178; del buf178  # reuse
    cpp_fused__softmax_54(c_void_p(buf181.data_ptr()), c_void_p(buf179.data_ptr()))
    buf182 = buf163; del buf163  # reuse
    # Source Nodes: [matmul_19], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf181, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf180, (6, 128, 64), (64, 384, 1), 0), out=buf182)
    buf183 = empty((1, 128, 6, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_55(c_void_p(buf182.data_ptr()), c_void_p(buf183.data_ptr()))
    buf184 = buf172; del buf172  # reuse
    # Source Nodes: [attn_output_19], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf183, (128, 384), (384, 1), 0), reinterpret_tensor(arg108_1, (384, 512), (1, 384), 0), out=buf184)
    del arg108_1
    buf185 = buf173; del buf173  # reuse
    buf186 = reinterpret_tensor(buf165, (1, 128, 512), (65536, 512, 1), 0); del buf165  # reuse
    cpp_fused_add_embedding_mean_mul_pow_rsqrt_56(c_void_p(arg192_1.data_ptr()), c_void_p(arg42_1.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf186.data_ptr()))
    del arg19_1
    buf187 = reinterpret_tensor(buf171, (128, 1024), (1024, 1), 0); del buf171  # reuse
    # Source Nodes: [l__mod___decoder_block_0_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf186, (128, 512), (512, 1), 0), reinterpret_tensor(arg109_1, (512, 1024), (1, 512), 0), out=buf187)
    del arg109_1
    buf188 = buf170; del buf170  # reuse
    # Source Nodes: [hidden_linear_8], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf186, (128, 512), (512, 1), 0), reinterpret_tensor(arg110_1, (512, 1024), (1, 512), 0), out=buf188)
    del arg110_1
    buf189 = reinterpret_tensor(buf187, (1, 128, 1024), (131072, 1024, 1), 0); del buf187  # reuse
    cpp_fused_add_mul_pow_tanh_57(c_void_p(buf189.data_ptr()), c_void_p(buf188.data_ptr()))
    buf190 = reinterpret_tensor(buf186, (128, 512), (512, 1), 0); del buf186  # reuse
    # Source Nodes: [forwarded_states_17], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf189, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg111_1, (1024, 512), (1, 1024), 0), out=buf190)
    del arg111_1
    buf191 = reinterpret_tensor(buf13, (1, 128, 512), (65536, 512, 1), 0); del buf13  # reuse
    buf192 = buf185; del buf185  # reuse
    buf193 = reinterpret_tensor(buf152, (1, 128, 512), (65536, 512, 1), 0); del buf152  # reuse
    cpp_fused_add_embedding_mean_mul_pow_rsqrt_58(c_void_p(buf191.data_ptr()), c_void_p(arg192_1.data_ptr()), c_void_p(arg42_1.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf193.data_ptr()))
    del arg192_1
    del arg20_1
    del arg42_1
    buf194 = reinterpret_tensor(buf183, (128, 384), (384, 1), 0); del buf183  # reuse
    # Source Nodes: [l__mod___decoder_block_1_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf193, (128, 512), (512, 1), 0), reinterpret_tensor(arg112_1, (512, 384), (1, 512), 0), out=buf194)
    del arg112_1
    buf195 = reinterpret_tensor(buf182, (128, 384), (384, 1), 0); del buf182  # reuse
    # Source Nodes: [l__mod___decoder_block_1_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf193, (128, 512), (512, 1), 0), reinterpret_tensor(arg113_1, (512, 384), (1, 512), 0), out=buf195)
    del arg113_1
    buf196 = reinterpret_tensor(buf181, (6, 128, 128), (16384, 128, 1), 0); del buf181  # reuse
    # Source Nodes: [scores_20], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf194, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf195, (6, 64, 128), (64, 1, 384), 0), out=buf196)
    buf197 = buf179; del buf179  # reuse
    buf198 = reinterpret_tensor(buf196, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf196  # reuse
    buf199 = buf198; del buf198  # reuse
    buf200 = buf177; del buf177  # reuse
    cpp_fused__softmax_59(c_void_p(buf199.data_ptr()), c_void_p(arg103_1.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf200.data_ptr()))
    buf201 = buf194; del buf194  # reuse
    # Source Nodes: [l__mod___decoder_block_1_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf193, (128, 512), (512, 1), 0), reinterpret_tensor(arg114_1, (512, 384), (1, 512), 0), out=buf201)
    del arg114_1
    buf202 = buf199; del buf199  # reuse
    cpp_fused__softmax_60(c_void_p(buf202.data_ptr()), c_void_p(buf200.data_ptr()))
    buf203 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_21], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf202, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf201, (6, 128, 64), (64, 384, 1), 0), out=buf203)
    buf204 = empty((1, 128, 6, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_61(c_void_p(buf203.data_ptr()), c_void_p(buf204.data_ptr()))
    buf205 = reinterpret_tensor(buf193, (128, 512), (512, 1), 0); del buf193  # reuse
    # Source Nodes: [attn_output_21], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf204, (128, 384), (384, 1), 0), reinterpret_tensor(arg115_1, (384, 512), (1, 384), 0), out=buf205)
    del arg115_1
    buf206 = buf192; del buf192  # reuse
    buf207 = reinterpret_tensor(buf190, (1, 128, 512), (65536, 512, 1), 0); del buf190  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_62(c_void_p(buf191.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf207.data_ptr()))
    del arg21_1
    buf208 = reinterpret_tensor(buf204, (128, 384), (384, 1), 0); del buf204  # reuse
    # Source Nodes: [l__mod___decoder_block_1_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf207, (128, 512), (512, 1), 0), reinterpret_tensor(arg116_1, (512, 384), (1, 512), 0), out=buf208)
    del arg116_1
    buf209 = reinterpret_tensor(buf203, (128, 384), (384, 1), 0); del buf203  # reuse
    # Source Nodes: [l__mod___decoder_block_1_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf174, (128, 512), (512, 1), 0), reinterpret_tensor(arg117_1, (512, 384), (1, 512), 0), out=buf209)
    del arg117_1
    buf210 = reinterpret_tensor(buf202, (6, 128, 128), (16384, 128, 1), 0); del buf202  # reuse
    # Source Nodes: [scores_22], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf208, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf209, (6, 64, 128), (64, 1, 384), 0), out=buf210)
    buf211 = buf200; del buf200  # reuse
    buf212 = reinterpret_tensor(buf210, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf210  # reuse
    buf213 = buf197; del buf197  # reuse
    cpp_fused__softmax_63(c_void_p(buf212.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf213.data_ptr()))
    buf214 = buf208; del buf208  # reuse
    # Source Nodes: [l__mod___decoder_block_1_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf174, (128, 512), (512, 1), 0), reinterpret_tensor(arg118_1, (512, 384), (1, 512), 0), out=buf214)
    del arg118_1
    buf215 = buf212; del buf212  # reuse
    cpp_fused__softmax_64(c_void_p(buf215.data_ptr()), c_void_p(buf213.data_ptr()))
    buf216 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_23], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf215, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf214, (6, 128, 64), (64, 384, 1), 0), out=buf216)
    buf217 = empty((1, 128, 6, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_65(c_void_p(buf216.data_ptr()), c_void_p(buf217.data_ptr()))
    buf218 = reinterpret_tensor(buf207, (128, 512), (512, 1), 0); del buf207  # reuse
    # Source Nodes: [attn_output_23], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf217, (128, 384), (384, 1), 0), reinterpret_tensor(arg119_1, (384, 512), (1, 384), 0), out=buf218)
    del arg119_1
    buf219 = buf206; del buf206  # reuse
    buf220 = reinterpret_tensor(buf184, (1, 128, 512), (65536, 512, 1), 0); del buf184  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_66(c_void_p(buf191.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf220.data_ptr()))
    del arg22_1
    buf221 = reinterpret_tensor(buf189, (128, 1024), (1024, 1), 0); del buf189  # reuse
    # Source Nodes: [l__mod___decoder_block_1_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf220, (128, 512), (512, 1), 0), reinterpret_tensor(arg120_1, (512, 1024), (1, 512), 0), out=buf221)
    del arg120_1
    buf222 = buf188; del buf188  # reuse
    # Source Nodes: [hidden_linear_9], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf220, (128, 512), (512, 1), 0), reinterpret_tensor(arg121_1, (512, 1024), (1, 512), 0), out=buf222)
    del arg121_1
    buf223 = reinterpret_tensor(buf221, (1, 128, 1024), (131072, 1024, 1), 0); del buf221  # reuse
    cpp_fused_add_mul_pow_tanh_67(c_void_p(buf223.data_ptr()), c_void_p(buf222.data_ptr()))
    buf224 = reinterpret_tensor(buf220, (128, 512), (512, 1), 0); del buf220  # reuse
    # Source Nodes: [forwarded_states_19], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf223, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg122_1, (1024, 512), (1, 1024), 0), out=buf224)
    del arg122_1
    buf225 = buf219; del buf219  # reuse
    buf226 = reinterpret_tensor(buf146, (1, 128, 512), (65536, 512, 1), 0); del buf146  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_68(c_void_p(buf191.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(arg23_1.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf226.data_ptr()))
    del arg23_1
    buf227 = reinterpret_tensor(buf217, (128, 384), (384, 1), 0); del buf217  # reuse
    # Source Nodes: [l__mod___decoder_block_2_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf226, (128, 512), (512, 1), 0), reinterpret_tensor(arg123_1, (512, 384), (1, 512), 0), out=buf227)
    del arg123_1
    buf228 = reinterpret_tensor(buf216, (128, 384), (384, 1), 0); del buf216  # reuse
    # Source Nodes: [l__mod___decoder_block_2_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf226, (128, 512), (512, 1), 0), reinterpret_tensor(arg124_1, (512, 384), (1, 512), 0), out=buf228)
    del arg124_1
    buf229 = reinterpret_tensor(buf215, (6, 128, 128), (16384, 128, 1), 0); del buf215  # reuse
    # Source Nodes: [scores_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf227, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf228, (6, 64, 128), (64, 1, 384), 0), out=buf229)
    buf230 = buf213; del buf213  # reuse
    buf231 = reinterpret_tensor(buf229, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf229  # reuse
    buf232 = buf231; del buf231  # reuse
    buf233 = buf211; del buf211  # reuse
    cpp_fused__softmax_69(c_void_p(buf232.data_ptr()), c_void_p(arg103_1.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf233.data_ptr()))
    buf234 = buf227; del buf227  # reuse
    # Source Nodes: [l__mod___decoder_block_2_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf226, (128, 512), (512, 1), 0), reinterpret_tensor(arg125_1, (512, 384), (1, 512), 0), out=buf234)
    del arg125_1
    buf235 = buf232; del buf232  # reuse
    cpp_fused__softmax_70(c_void_p(buf235.data_ptr()), c_void_p(buf233.data_ptr()))
    buf236 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_25], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf235, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf234, (6, 128, 64), (64, 384, 1), 0), out=buf236)
    buf237 = empty((1, 128, 6, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_71(c_void_p(buf236.data_ptr()), c_void_p(buf237.data_ptr()))
    buf238 = reinterpret_tensor(buf226, (128, 512), (512, 1), 0); del buf226  # reuse
    # Source Nodes: [attn_output_25], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf237, (128, 384), (384, 1), 0), reinterpret_tensor(arg126_1, (384, 512), (1, 384), 0), out=buf238)
    del arg126_1
    buf239 = buf191; del buf191  # reuse
    buf240 = buf225; del buf225  # reuse
    buf241 = reinterpret_tensor(buf133, (1, 128, 512), (65536, 512, 1), 0); del buf133  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_72(c_void_p(buf239.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(arg24_1.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf241.data_ptr()))
    del arg24_1
    buf242 = reinterpret_tensor(buf237, (128, 384), (384, 1), 0); del buf237  # reuse
    # Source Nodes: [l__mod___decoder_block_2_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf241, (128, 512), (512, 1), 0), reinterpret_tensor(arg127_1, (512, 384), (1, 512), 0), out=buf242)
    del arg127_1
    buf243 = reinterpret_tensor(buf236, (128, 384), (384, 1), 0); del buf236  # reuse
    # Source Nodes: [l__mod___decoder_block_2_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf174, (128, 512), (512, 1), 0), reinterpret_tensor(arg128_1, (512, 384), (1, 512), 0), out=buf243)
    del arg128_1
    buf244 = reinterpret_tensor(buf235, (6, 128, 128), (16384, 128, 1), 0); del buf235  # reuse
    # Source Nodes: [scores_26], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf242, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf243, (6, 64, 128), (64, 1, 384), 0), out=buf244)
    buf245 = buf233; del buf233  # reuse
    buf246 = reinterpret_tensor(buf244, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf244  # reuse
    buf247 = buf230; del buf230  # reuse
    cpp_fused__softmax_73(c_void_p(buf246.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf247.data_ptr()))
    buf248 = buf242; del buf242  # reuse
    # Source Nodes: [l__mod___decoder_block_2_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf174, (128, 512), (512, 1), 0), reinterpret_tensor(arg129_1, (512, 384), (1, 512), 0), out=buf248)
    del arg129_1
    buf249 = buf246; del buf246  # reuse
    cpp_fused__softmax_74(c_void_p(buf249.data_ptr()), c_void_p(buf247.data_ptr()))
    buf250 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_27], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf249, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf248, (6, 128, 64), (64, 384, 1), 0), out=buf250)
    buf251 = empty((1, 128, 6, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_75(c_void_p(buf250.data_ptr()), c_void_p(buf251.data_ptr()))
    buf252 = reinterpret_tensor(buf241, (128, 512), (512, 1), 0); del buf241  # reuse
    # Source Nodes: [attn_output_27], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf251, (128, 384), (384, 1), 0), reinterpret_tensor(arg130_1, (384, 512), (1, 384), 0), out=buf252)
    del arg130_1
    buf253 = buf240; del buf240  # reuse
    buf254 = reinterpret_tensor(buf238, (1, 128, 512), (65536, 512, 1), 0); del buf238  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_76(c_void_p(buf239.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(arg25_1.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf254.data_ptr()))
    del arg25_1
    buf255 = reinterpret_tensor(buf223, (128, 1024), (1024, 1), 0); del buf223  # reuse
    # Source Nodes: [l__mod___decoder_block_2_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf254, (128, 512), (512, 1), 0), reinterpret_tensor(arg131_1, (512, 1024), (1, 512), 0), out=buf255)
    del arg131_1
    buf256 = buf222; del buf222  # reuse
    # Source Nodes: [hidden_linear_10], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf254, (128, 512), (512, 1), 0), reinterpret_tensor(arg132_1, (512, 1024), (1, 512), 0), out=buf256)
    del arg132_1
    buf257 = reinterpret_tensor(buf255, (1, 128, 1024), (131072, 1024, 1), 0); del buf255  # reuse
    cpp_fused_add_mul_pow_tanh_77(c_void_p(buf257.data_ptr()), c_void_p(buf256.data_ptr()))
    buf258 = reinterpret_tensor(buf254, (128, 512), (512, 1), 0); del buf254  # reuse
    # Source Nodes: [forwarded_states_21], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf257, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg133_1, (1024, 512), (1, 1024), 0), out=buf258)
    del arg133_1
    buf259 = buf253; del buf253  # reuse
    buf260 = reinterpret_tensor(buf224, (1, 128, 512), (65536, 512, 1), 0); del buf224  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_78(c_void_p(buf239.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf260.data_ptr()))
    del arg26_1
    buf261 = reinterpret_tensor(buf251, (128, 384), (384, 1), 0); del buf251  # reuse
    # Source Nodes: [l__mod___decoder_block_3_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf260, (128, 512), (512, 1), 0), reinterpret_tensor(arg134_1, (512, 384), (1, 512), 0), out=buf261)
    del arg134_1
    buf262 = reinterpret_tensor(buf250, (128, 384), (384, 1), 0); del buf250  # reuse
    # Source Nodes: [l__mod___decoder_block_3_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf260, (128, 512), (512, 1), 0), reinterpret_tensor(arg135_1, (512, 384), (1, 512), 0), out=buf262)
    del arg135_1
    buf263 = reinterpret_tensor(buf249, (6, 128, 128), (16384, 128, 1), 0); del buf249  # reuse
    # Source Nodes: [scores_28], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf261, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf262, (6, 64, 128), (64, 1, 384), 0), out=buf263)
    buf264 = buf247; del buf247  # reuse
    buf265 = reinterpret_tensor(buf263, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf263  # reuse
    buf266 = buf265; del buf265  # reuse
    buf267 = buf245; del buf245  # reuse
    cpp_fused__softmax_79(c_void_p(buf266.data_ptr()), c_void_p(arg103_1.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf267.data_ptr()))
    buf268 = buf261; del buf261  # reuse
    # Source Nodes: [l__mod___decoder_block_3_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf260, (128, 512), (512, 1), 0), reinterpret_tensor(arg136_1, (512, 384), (1, 512), 0), out=buf268)
    del arg136_1
    buf269 = buf266; del buf266  # reuse
    cpp_fused__softmax_80(c_void_p(buf269.data_ptr()), c_void_p(buf267.data_ptr()))
    buf270 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_29], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf269, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf268, (6, 128, 64), (64, 384, 1), 0), out=buf270)
    buf271 = empty((1, 128, 6, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_81(c_void_p(buf270.data_ptr()), c_void_p(buf271.data_ptr()))
    buf272 = reinterpret_tensor(buf260, (128, 512), (512, 1), 0); del buf260  # reuse
    # Source Nodes: [attn_output_29], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf271, (128, 384), (384, 1), 0), reinterpret_tensor(arg137_1, (384, 512), (1, 384), 0), out=buf272)
    del arg137_1
    buf273 = buf259; del buf259  # reuse
    buf274 = reinterpret_tensor(buf218, (1, 128, 512), (65536, 512, 1), 0); del buf218  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_82(c_void_p(buf239.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(arg27_1.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf274.data_ptr()))
    del arg27_1
    buf275 = reinterpret_tensor(buf271, (128, 384), (384, 1), 0); del buf271  # reuse
    # Source Nodes: [l__mod___decoder_block_3_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf274, (128, 512), (512, 1), 0), reinterpret_tensor(arg138_1, (512, 384), (1, 512), 0), out=buf275)
    del arg138_1
    buf276 = reinterpret_tensor(buf270, (128, 384), (384, 1), 0); del buf270  # reuse
    # Source Nodes: [l__mod___decoder_block_3_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf174, (128, 512), (512, 1), 0), reinterpret_tensor(arg139_1, (512, 384), (1, 512), 0), out=buf276)
    del arg139_1
    buf277 = reinterpret_tensor(buf269, (6, 128, 128), (16384, 128, 1), 0); del buf269  # reuse
    # Source Nodes: [scores_30], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf275, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf276, (6, 64, 128), (64, 1, 384), 0), out=buf277)
    buf278 = buf267; del buf267  # reuse
    buf279 = reinterpret_tensor(buf277, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf277  # reuse
    buf280 = buf264; del buf264  # reuse
    cpp_fused__softmax_83(c_void_p(buf279.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf280.data_ptr()))
    buf281 = buf275; del buf275  # reuse
    # Source Nodes: [l__mod___decoder_block_3_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf174, (128, 512), (512, 1), 0), reinterpret_tensor(arg140_1, (512, 384), (1, 512), 0), out=buf281)
    del arg140_1
    buf282 = buf279; del buf279  # reuse
    cpp_fused__softmax_84(c_void_p(buf282.data_ptr()), c_void_p(buf280.data_ptr()))
    buf283 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_31], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf282, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf281, (6, 128, 64), (64, 384, 1), 0), out=buf283)
    buf284 = empty((1, 128, 6, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_85(c_void_p(buf283.data_ptr()), c_void_p(buf284.data_ptr()))
    buf285 = reinterpret_tensor(buf274, (128, 512), (512, 1), 0); del buf274  # reuse
    # Source Nodes: [attn_output_31], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf284, (128, 384), (384, 1), 0), reinterpret_tensor(arg141_1, (384, 512), (1, 384), 0), out=buf285)
    del arg141_1
    buf286 = buf239; del buf239  # reuse
    buf287 = buf273; del buf273  # reuse
    buf288 = reinterpret_tensor(buf205, (1, 128, 512), (65536, 512, 1), 0); del buf205  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_86(c_void_p(buf286.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf288.data_ptr()))
    del arg28_1
    buf289 = reinterpret_tensor(buf257, (128, 1024), (1024, 1), 0); del buf257  # reuse
    # Source Nodes: [l__mod___decoder_block_3_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf288, (128, 512), (512, 1), 0), reinterpret_tensor(arg142_1, (512, 1024), (1, 512), 0), out=buf289)
    del arg142_1
    buf290 = buf256; del buf256  # reuse
    # Source Nodes: [hidden_linear_11], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf288, (128, 512), (512, 1), 0), reinterpret_tensor(arg143_1, (512, 1024), (1, 512), 0), out=buf290)
    del arg143_1
    buf291 = reinterpret_tensor(buf289, (1, 128, 1024), (131072, 1024, 1), 0); del buf289  # reuse
    cpp_fused_add_mul_pow_tanh_87(c_void_p(buf291.data_ptr()), c_void_p(buf290.data_ptr()))
    buf292 = reinterpret_tensor(buf288, (128, 512), (512, 1), 0); del buf288  # reuse
    # Source Nodes: [forwarded_states_23], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf291, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg144_1, (1024, 512), (1, 1024), 0), out=buf292)
    del arg144_1
    buf293 = buf287; del buf287  # reuse
    buf294 = reinterpret_tensor(buf285, (1, 128, 512), (65536, 512, 1), 0); del buf285  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_88(c_void_p(buf286.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(arg29_1.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()))
    del arg29_1
    buf295 = reinterpret_tensor(buf284, (128, 384), (384, 1), 0); del buf284  # reuse
    # Source Nodes: [l__mod___decoder_block_4_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf294, (128, 512), (512, 1), 0), reinterpret_tensor(arg145_1, (512, 384), (1, 512), 0), out=buf295)
    del arg145_1
    buf296 = reinterpret_tensor(buf283, (128, 384), (384, 1), 0); del buf283  # reuse
    # Source Nodes: [l__mod___decoder_block_4_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf294, (128, 512), (512, 1), 0), reinterpret_tensor(arg146_1, (512, 384), (1, 512), 0), out=buf296)
    del arg146_1
    buf297 = reinterpret_tensor(buf282, (6, 128, 128), (16384, 128, 1), 0); del buf282  # reuse
    # Source Nodes: [scores_32], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf295, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf296, (6, 64, 128), (64, 1, 384), 0), out=buf297)
    buf298 = buf280; del buf280  # reuse
    buf299 = reinterpret_tensor(buf297, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf297  # reuse
    buf300 = buf299; del buf299  # reuse
    buf301 = buf278; del buf278  # reuse
    cpp_fused__softmax_89(c_void_p(buf300.data_ptr()), c_void_p(arg103_1.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf301.data_ptr()))
    buf302 = buf295; del buf295  # reuse
    # Source Nodes: [l__mod___decoder_block_4_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf294, (128, 512), (512, 1), 0), reinterpret_tensor(arg147_1, (512, 384), (1, 512), 0), out=buf302)
    del arg147_1
    buf303 = buf300; del buf300  # reuse
    cpp_fused__softmax_90(c_void_p(buf303.data_ptr()), c_void_p(buf301.data_ptr()))
    buf304 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_33], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf303, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf302, (6, 128, 64), (64, 384, 1), 0), out=buf304)
    buf305 = empty((1, 128, 6, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_91(c_void_p(buf304.data_ptr()), c_void_p(buf305.data_ptr()))
    buf306 = reinterpret_tensor(buf294, (128, 512), (512, 1), 0); del buf294  # reuse
    # Source Nodes: [attn_output_33], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf305, (128, 384), (384, 1), 0), reinterpret_tensor(arg148_1, (384, 512), (1, 384), 0), out=buf306)
    del arg148_1
    buf307 = buf293; del buf293  # reuse
    buf308 = reinterpret_tensor(buf272, (1, 128, 512), (65536, 512, 1), 0); del buf272  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_92(c_void_p(buf286.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf308.data_ptr()))
    del arg30_1
    buf309 = reinterpret_tensor(buf305, (128, 384), (384, 1), 0); del buf305  # reuse
    # Source Nodes: [l__mod___decoder_block_4_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf308, (128, 512), (512, 1), 0), reinterpret_tensor(arg149_1, (512, 384), (1, 512), 0), out=buf309)
    del arg149_1
    buf310 = reinterpret_tensor(buf304, (128, 384), (384, 1), 0); del buf304  # reuse
    # Source Nodes: [l__mod___decoder_block_4_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf174, (128, 512), (512, 1), 0), reinterpret_tensor(arg150_1, (512, 384), (1, 512), 0), out=buf310)
    del arg150_1
    buf311 = reinterpret_tensor(buf303, (6, 128, 128), (16384, 128, 1), 0); del buf303  # reuse
    # Source Nodes: [scores_34], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf309, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf310, (6, 64, 128), (64, 1, 384), 0), out=buf311)
    buf312 = buf301; del buf301  # reuse
    buf313 = reinterpret_tensor(buf311, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf311  # reuse
    buf314 = buf298; del buf298  # reuse
    cpp_fused__softmax_93(c_void_p(buf313.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf314.data_ptr()))
    buf315 = buf309; del buf309  # reuse
    # Source Nodes: [l__mod___decoder_block_4_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf174, (128, 512), (512, 1), 0), reinterpret_tensor(arg151_1, (512, 384), (1, 512), 0), out=buf315)
    del arg151_1
    buf316 = buf313; del buf313  # reuse
    cpp_fused__softmax_94(c_void_p(buf316.data_ptr()), c_void_p(buf314.data_ptr()))
    buf317 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_35], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf316, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf315, (6, 128, 64), (64, 384, 1), 0), out=buf317)
    buf318 = empty((1, 128, 6, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_95(c_void_p(buf317.data_ptr()), c_void_p(buf318.data_ptr()))
    buf319 = reinterpret_tensor(buf308, (128, 512), (512, 1), 0); del buf308  # reuse
    # Source Nodes: [attn_output_35], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf318, (128, 384), (384, 1), 0), reinterpret_tensor(arg152_1, (384, 512), (1, 384), 0), out=buf319)
    del arg152_1
    buf320 = buf307; del buf307  # reuse
    buf321 = reinterpret_tensor(buf258, (1, 128, 512), (65536, 512, 1), 0); del buf258  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_96(c_void_p(buf286.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(arg31_1.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf321.data_ptr()))
    del arg31_1
    buf322 = reinterpret_tensor(buf291, (128, 1024), (1024, 1), 0); del buf291  # reuse
    # Source Nodes: [l__mod___decoder_block_4_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf321, (128, 512), (512, 1), 0), reinterpret_tensor(arg153_1, (512, 1024), (1, 512), 0), out=buf322)
    del arg153_1
    buf323 = buf290; del buf290  # reuse
    # Source Nodes: [hidden_linear_12], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf321, (128, 512), (512, 1), 0), reinterpret_tensor(arg154_1, (512, 1024), (1, 512), 0), out=buf323)
    del arg154_1
    buf324 = reinterpret_tensor(buf322, (1, 128, 1024), (131072, 1024, 1), 0); del buf322  # reuse
    cpp_fused_add_mul_pow_tanh_97(c_void_p(buf324.data_ptr()), c_void_p(buf323.data_ptr()))
    buf325 = reinterpret_tensor(buf321, (128, 512), (512, 1), 0); del buf321  # reuse
    # Source Nodes: [forwarded_states_25], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf324, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg155_1, (1024, 512), (1, 1024), 0), out=buf325)
    del arg155_1
    buf326 = buf286; del buf286  # reuse
    buf327 = buf320; del buf320  # reuse
    buf328 = reinterpret_tensor(buf252, (1, 128, 512), (65536, 512, 1), 0); del buf252  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_98(c_void_p(buf326.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf328.data_ptr()))
    del arg32_1
    buf329 = reinterpret_tensor(buf318, (128, 384), (384, 1), 0); del buf318  # reuse
    # Source Nodes: [l__mod___decoder_block_5_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf328, (128, 512), (512, 1), 0), reinterpret_tensor(arg156_1, (512, 384), (1, 512), 0), out=buf329)
    del arg156_1
    buf330 = reinterpret_tensor(buf317, (128, 384), (384, 1), 0); del buf317  # reuse
    # Source Nodes: [l__mod___decoder_block_5_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf328, (128, 512), (512, 1), 0), reinterpret_tensor(arg157_1, (512, 384), (1, 512), 0), out=buf330)
    del arg157_1
    buf331 = reinterpret_tensor(buf316, (6, 128, 128), (16384, 128, 1), 0); del buf316  # reuse
    # Source Nodes: [scores_36], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf329, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf330, (6, 64, 128), (64, 1, 384), 0), out=buf331)
    buf332 = buf314; del buf314  # reuse
    buf333 = reinterpret_tensor(buf331, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf331  # reuse
    buf334 = buf333; del buf333  # reuse
    buf335 = buf312; del buf312  # reuse
    cpp_fused__softmax_99(c_void_p(buf334.data_ptr()), c_void_p(arg103_1.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf335.data_ptr()))
    buf336 = buf329; del buf329  # reuse
    # Source Nodes: [l__mod___decoder_block_5_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf328, (128, 512), (512, 1), 0), reinterpret_tensor(arg158_1, (512, 384), (1, 512), 0), out=buf336)
    del arg158_1
    buf337 = buf334; del buf334  # reuse
    cpp_fused__softmax_100(c_void_p(buf337.data_ptr()), c_void_p(buf335.data_ptr()))
    buf338 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_37], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf337, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf336, (6, 128, 64), (64, 384, 1), 0), out=buf338)
    buf339 = empty((1, 128, 6, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_101(c_void_p(buf338.data_ptr()), c_void_p(buf339.data_ptr()))
    buf340 = reinterpret_tensor(buf328, (128, 512), (512, 1), 0); del buf328  # reuse
    # Source Nodes: [attn_output_37], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf339, (128, 384), (384, 1), 0), reinterpret_tensor(arg159_1, (384, 512), (1, 384), 0), out=buf340)
    del arg159_1
    buf341 = buf327; del buf327  # reuse
    buf342 = reinterpret_tensor(buf325, (1, 128, 512), (65536, 512, 1), 0); del buf325  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_102(c_void_p(buf326.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(arg33_1.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf342.data_ptr()))
    del arg33_1
    buf343 = reinterpret_tensor(buf339, (128, 384), (384, 1), 0); del buf339  # reuse
    # Source Nodes: [l__mod___decoder_block_5_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf342, (128, 512), (512, 1), 0), reinterpret_tensor(arg160_1, (512, 384), (1, 512), 0), out=buf343)
    del arg160_1
    buf344 = reinterpret_tensor(buf338, (128, 384), (384, 1), 0); del buf338  # reuse
    # Source Nodes: [l__mod___decoder_block_5_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf174, (128, 512), (512, 1), 0), reinterpret_tensor(arg161_1, (512, 384), (1, 512), 0), out=buf344)
    del arg161_1
    buf345 = reinterpret_tensor(buf337, (6, 128, 128), (16384, 128, 1), 0); del buf337  # reuse
    # Source Nodes: [scores_38], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf343, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf344, (6, 64, 128), (64, 1, 384), 0), out=buf345)
    buf346 = buf335; del buf335  # reuse
    buf347 = reinterpret_tensor(buf345, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf345  # reuse
    buf348 = buf332; del buf332  # reuse
    cpp_fused__softmax_103(c_void_p(buf347.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(buf348.data_ptr()))
    buf349 = buf343; del buf343  # reuse
    # Source Nodes: [l__mod___decoder_block_5_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf174, (128, 512), (512, 1), 0), reinterpret_tensor(arg162_1, (512, 384), (1, 512), 0), out=buf349)
    del arg162_1
    buf350 = buf347; del buf347  # reuse
    cpp_fused__softmax_104(c_void_p(buf350.data_ptr()), c_void_p(buf348.data_ptr()))
    buf351 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_39], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf350, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf349, (6, 128, 64), (64, 384, 1), 0), out=buf351)
    buf352 = empty((1, 128, 6, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_105(c_void_p(buf351.data_ptr()), c_void_p(buf352.data_ptr()))
    buf353 = reinterpret_tensor(buf342, (128, 512), (512, 1), 0); del buf342  # reuse
    # Source Nodes: [attn_output_39], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf352, (128, 384), (384, 1), 0), reinterpret_tensor(arg163_1, (384, 512), (1, 384), 0), out=buf353)
    del arg163_1
    buf354 = buf341; del buf341  # reuse
    buf355 = reinterpret_tensor(buf319, (1, 128, 512), (65536, 512, 1), 0); del buf319  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_106(c_void_p(buf326.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf355.data_ptr()))
    del arg34_1
    buf356 = reinterpret_tensor(buf324, (128, 1024), (1024, 1), 0); del buf324  # reuse
    # Source Nodes: [l__mod___decoder_block_5_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf355, (128, 512), (512, 1), 0), reinterpret_tensor(arg164_1, (512, 1024), (1, 512), 0), out=buf356)
    del arg164_1
    buf357 = buf323; del buf323  # reuse
    # Source Nodes: [hidden_linear_13], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf355, (128, 512), (512, 1), 0), reinterpret_tensor(arg165_1, (512, 1024), (1, 512), 0), out=buf357)
    del arg165_1
    buf358 = reinterpret_tensor(buf356, (1, 128, 1024), (131072, 1024, 1), 0); del buf356  # reuse
    cpp_fused_add_mul_pow_tanh_107(c_void_p(buf358.data_ptr()), c_void_p(buf357.data_ptr()))
    buf359 = reinterpret_tensor(buf355, (128, 512), (512, 1), 0); del buf355  # reuse
    # Source Nodes: [forwarded_states_27], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf358, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg166_1, (1024, 512), (1, 1024), 0), out=buf359)
    del arg166_1
    buf360 = buf354; del buf354  # reuse
    buf361 = reinterpret_tensor(buf306, (1, 128, 512), (65536, 512, 1), 0); del buf306  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_108(c_void_p(buf326.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(arg35_1.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf361.data_ptr()))
    del arg35_1
    buf362 = reinterpret_tensor(buf352, (128, 384), (384, 1), 0); del buf352  # reuse
    # Source Nodes: [l__mod___decoder_block_6_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf361, (128, 512), (512, 1), 0), reinterpret_tensor(arg167_1, (512, 384), (1, 512), 0), out=buf362)
    del arg167_1
    buf363 = reinterpret_tensor(buf351, (128, 384), (384, 1), 0); del buf351  # reuse
    # Source Nodes: [l__mod___decoder_block_6_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf361, (128, 512), (512, 1), 0), reinterpret_tensor(arg168_1, (512, 384), (1, 512), 0), out=buf363)
    del arg168_1
    buf364 = reinterpret_tensor(buf350, (6, 128, 128), (16384, 128, 1), 0); del buf350  # reuse
    # Source Nodes: [scores_40], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf362, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf363, (6, 64, 128), (64, 1, 384), 0), out=buf364)
    buf365 = buf348; del buf348  # reuse
    buf366 = reinterpret_tensor(buf364, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf364  # reuse
    buf367 = buf366; del buf366  # reuse
    buf368 = buf346; del buf346  # reuse
    cpp_fused__softmax_109(c_void_p(buf367.data_ptr()), c_void_p(arg103_1.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(buf368.data_ptr()))
    buf369 = buf362; del buf362  # reuse
    # Source Nodes: [l__mod___decoder_block_6_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf361, (128, 512), (512, 1), 0), reinterpret_tensor(arg169_1, (512, 384), (1, 512), 0), out=buf369)
    del arg169_1
    buf370 = buf367; del buf367  # reuse
    cpp_fused__softmax_110(c_void_p(buf370.data_ptr()), c_void_p(buf368.data_ptr()))
    buf371 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_41], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf370, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf369, (6, 128, 64), (64, 384, 1), 0), out=buf371)
    buf372 = empty((1, 128, 6, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_111(c_void_p(buf371.data_ptr()), c_void_p(buf372.data_ptr()))
    buf373 = reinterpret_tensor(buf361, (128, 512), (512, 1), 0); del buf361  # reuse
    # Source Nodes: [attn_output_41], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf372, (128, 384), (384, 1), 0), reinterpret_tensor(arg170_1, (384, 512), (1, 384), 0), out=buf373)
    del arg170_1
    buf374 = buf326; del buf326  # reuse
    buf375 = buf360; del buf360  # reuse
    buf376 = reinterpret_tensor(buf292, (1, 128, 512), (65536, 512, 1), 0); del buf292  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_112(c_void_p(buf374.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf376.data_ptr()))
    del arg36_1
    buf377 = reinterpret_tensor(buf372, (128, 384), (384, 1), 0); del buf372  # reuse
    # Source Nodes: [l__mod___decoder_block_6_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf376, (128, 512), (512, 1), 0), reinterpret_tensor(arg171_1, (512, 384), (1, 512), 0), out=buf377)
    del arg171_1
    buf378 = reinterpret_tensor(buf371, (128, 384), (384, 1), 0); del buf371  # reuse
    # Source Nodes: [l__mod___decoder_block_6_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf174, (128, 512), (512, 1), 0), reinterpret_tensor(arg172_1, (512, 384), (1, 512), 0), out=buf378)
    del arg172_1
    buf379 = reinterpret_tensor(buf370, (6, 128, 128), (16384, 128, 1), 0); del buf370  # reuse
    # Source Nodes: [scores_42], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf377, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf378, (6, 64, 128), (64, 1, 384), 0), out=buf379)
    buf380 = buf368; del buf368  # reuse
    buf381 = reinterpret_tensor(buf379, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf379  # reuse
    buf382 = buf365; del buf365  # reuse
    cpp_fused__softmax_113(c_void_p(buf381.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf382.data_ptr()))
    buf383 = buf377; del buf377  # reuse
    # Source Nodes: [l__mod___decoder_block_6_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf174, (128, 512), (512, 1), 0), reinterpret_tensor(arg173_1, (512, 384), (1, 512), 0), out=buf383)
    del arg173_1
    buf384 = buf381; del buf381  # reuse
    cpp_fused__softmax_114(c_void_p(buf384.data_ptr()), c_void_p(buf382.data_ptr()))
    buf385 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_43], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf384, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf383, (6, 128, 64), (64, 384, 1), 0), out=buf385)
    buf386 = empty((1, 128, 6, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_115(c_void_p(buf385.data_ptr()), c_void_p(buf386.data_ptr()))
    buf387 = reinterpret_tensor(buf376, (128, 512), (512, 1), 0); del buf376  # reuse
    # Source Nodes: [attn_output_43], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf386, (128, 384), (384, 1), 0), reinterpret_tensor(arg174_1, (384, 512), (1, 384), 0), out=buf387)
    del arg174_1
    buf388 = buf375; del buf375  # reuse
    buf389 = reinterpret_tensor(buf373, (1, 128, 512), (65536, 512, 1), 0); del buf373  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_116(c_void_p(buf374.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(arg37_1.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf389.data_ptr()))
    del arg37_1
    buf390 = reinterpret_tensor(buf358, (128, 1024), (1024, 1), 0); del buf358  # reuse
    # Source Nodes: [l__mod___decoder_block_6_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf389, (128, 512), (512, 1), 0), reinterpret_tensor(arg175_1, (512, 1024), (1, 512), 0), out=buf390)
    del arg175_1
    buf391 = buf357; del buf357  # reuse
    # Source Nodes: [hidden_linear_14], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf389, (128, 512), (512, 1), 0), reinterpret_tensor(arg176_1, (512, 1024), (1, 512), 0), out=buf391)
    del arg176_1
    buf392 = reinterpret_tensor(buf390, (1, 128, 1024), (131072, 1024, 1), 0); del buf390  # reuse
    cpp_fused_add_mul_pow_tanh_117(c_void_p(buf392.data_ptr()), c_void_p(buf391.data_ptr()))
    buf393 = reinterpret_tensor(buf389, (128, 512), (512, 1), 0); del buf389  # reuse
    # Source Nodes: [forwarded_states_29], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf392, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg177_1, (1024, 512), (1, 1024), 0), out=buf393)
    del arg177_1
    buf394 = buf388; del buf388  # reuse
    buf395 = reinterpret_tensor(buf359, (1, 128, 512), (65536, 512, 1), 0); del buf359  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_118(c_void_p(buf374.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(buf395.data_ptr()))
    del arg38_1
    buf396 = reinterpret_tensor(buf386, (128, 384), (384, 1), 0); del buf386  # reuse
    # Source Nodes: [l__mod___decoder_block_7_layer_0_self_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf395, (128, 512), (512, 1), 0), reinterpret_tensor(arg178_1, (512, 384), (1, 512), 0), out=buf396)
    del arg178_1
    buf397 = reinterpret_tensor(buf385, (128, 384), (384, 1), 0); del buf385  # reuse
    # Source Nodes: [l__mod___decoder_block_7_layer_0_self_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf395, (128, 512), (512, 1), 0), reinterpret_tensor(arg179_1, (512, 384), (1, 512), 0), out=buf397)
    del arg179_1
    buf398 = reinterpret_tensor(buf384, (6, 128, 128), (16384, 128, 1), 0); del buf384  # reuse
    # Source Nodes: [scores_44], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf396, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf397, (6, 64, 128), (64, 1, 384), 0), out=buf398)
    buf399 = buf382; del buf382  # reuse
    buf400 = reinterpret_tensor(buf398, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf398  # reuse
    buf401 = buf400; del buf400  # reuse
    buf402 = buf380; del buf380  # reuse
    cpp_fused__softmax_119(c_void_p(buf401.data_ptr()), c_void_p(arg103_1.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf402.data_ptr()))
    del arg103_1
    buf403 = buf396; del buf396  # reuse
    # Source Nodes: [l__mod___decoder_block_7_layer_0_self_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf395, (128, 512), (512, 1), 0), reinterpret_tensor(arg180_1, (512, 384), (1, 512), 0), out=buf403)
    del arg180_1
    buf404 = buf401; del buf401  # reuse
    cpp_fused__softmax_120(c_void_p(buf404.data_ptr()), c_void_p(buf402.data_ptr()))
    buf405 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_45], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf404, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf403, (6, 128, 64), (64, 384, 1), 0), out=buf405)
    buf406 = empty((1, 128, 6, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_121(c_void_p(buf405.data_ptr()), c_void_p(buf406.data_ptr()))
    buf407 = reinterpret_tensor(buf395, (128, 512), (512, 1), 0); del buf395  # reuse
    # Source Nodes: [attn_output_45], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf406, (128, 384), (384, 1), 0), reinterpret_tensor(arg181_1, (384, 512), (1, 384), 0), out=buf407)
    del arg181_1
    buf408 = buf394; del buf394  # reuse
    buf409 = reinterpret_tensor(buf353, (1, 128, 512), (65536, 512, 1), 0); del buf353  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_122(c_void_p(buf374.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(arg39_1.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(buf409.data_ptr()))
    del arg39_1
    buf410 = reinterpret_tensor(buf406, (128, 384), (384, 1), 0); del buf406  # reuse
    # Source Nodes: [l__mod___decoder_block_7_layer_1_enc_dec_attention_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf409, (128, 512), (512, 1), 0), reinterpret_tensor(arg182_1, (512, 384), (1, 512), 0), out=buf410)
    del arg182_1
    buf411 = reinterpret_tensor(buf405, (128, 384), (384, 1), 0); del buf405  # reuse
    # Source Nodes: [l__mod___decoder_block_7_layer_1_enc_dec_attention_k], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf174, (128, 512), (512, 1), 0), reinterpret_tensor(arg183_1, (512, 384), (1, 512), 0), out=buf411)
    del arg183_1
    buf412 = reinterpret_tensor(buf404, (6, 128, 128), (16384, 128, 1), 0); del buf404  # reuse
    # Source Nodes: [scores_46], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf410, (6, 128, 64), (64, 384, 1), 0), reinterpret_tensor(buf411, (6, 64, 128), (64, 1, 384), 0), out=buf412)
    buf413 = buf402; del buf402  # reuse
    buf414 = reinterpret_tensor(buf412, (1, 6, 128, 128), (98304, 16384, 128, 1), 0); del buf412  # reuse
    buf415 = buf399; del buf399  # reuse
    cpp_fused__softmax_123(c_void_p(buf414.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(buf415.data_ptr()))
    del buf413
    buf416 = buf410; del buf410  # reuse
    # Source Nodes: [l__mod___decoder_block_7_layer_1_enc_dec_attention_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf174, (128, 512), (512, 1), 0), reinterpret_tensor(arg184_1, (512, 384), (1, 512), 0), out=buf416)
    del arg184_1
    buf417 = buf414; del buf414  # reuse
    cpp_fused__softmax_124(c_void_p(buf417.data_ptr()), c_void_p(buf415.data_ptr()))
    del buf415
    buf418 = empty((6, 128, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_47], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf417, (6, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf416, (6, 128, 64), (64, 384, 1), 0), out=buf418)
    del buf417
    buf419 = empty((1, 128, 6, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_125(c_void_p(buf418.data_ptr()), c_void_p(buf419.data_ptr()))
    del buf418
    buf420 = reinterpret_tensor(buf409, (128, 512), (512, 1), 0); del buf409  # reuse
    # Source Nodes: [attn_output_47], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf419, (128, 384), (384, 1), 0), reinterpret_tensor(arg185_1, (384, 512), (1, 384), 0), out=buf420)
    del arg185_1
    del buf419
    buf421 = buf374; del buf374  # reuse
    buf422 = buf408; del buf408  # reuse
    buf423 = reinterpret_tensor(buf340, (1, 128, 512), (65536, 512, 1), 0); del buf340  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_126(c_void_p(buf421.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(buf423.data_ptr()))
    del arg40_1
    del buf387
    del buf393
    del buf407
    del buf420
    buf424 = reinterpret_tensor(buf392, (128, 1024), (1024, 1), 0); del buf392  # reuse
    # Source Nodes: [l__mod___decoder_block_7_layer__1__dense_relu_dense_wi_0], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf423, (128, 512), (512, 1), 0), reinterpret_tensor(arg186_1, (512, 1024), (1, 512), 0), out=buf424)
    del arg186_1
    buf425 = buf391; del buf391  # reuse
    # Source Nodes: [hidden_linear_15], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf423, (128, 512), (512, 1), 0), reinterpret_tensor(arg187_1, (512, 1024), (1, 512), 0), out=buf425)
    del arg187_1
    buf426 = reinterpret_tensor(buf424, (1, 128, 1024), (131072, 1024, 1), 0); del buf424  # reuse
    cpp_fused_add_mul_pow_tanh_127(c_void_p(buf426.data_ptr()), c_void_p(buf425.data_ptr()))
    del buf425
    buf427 = reinterpret_tensor(buf423, (128, 512), (512, 1), 0); del buf423  # reuse
    # Source Nodes: [forwarded_states_31], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf426, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg188_1, (1024, 512), (1, 1024), 0), out=buf427)
    del arg188_1
    del buf426
    buf428 = buf422; del buf422  # reuse
    buf429 = buf421; del buf421  # reuse
    cpp_fused_add_mean_mul_pow_rsqrt_128(c_void_p(buf429.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(arg41_1.data_ptr()), c_void_p(buf428.data_ptr()))
    del arg41_1
    del buf427
    buf430 = empty((128, 250112), device='cpu', dtype=torch.float32)
    # Source Nodes: [lm_logits], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf429, (128, 512), (512, 1), 0), reinterpret_tensor(arg189_1, (512, 250112), (1, 512), 0), out=buf430)
    del arg189_1
    del buf429
    buf431 = reinterpret_tensor(buf428, (128, 1), (1, 128), 0); del buf428  # reuse
    buf432 = reinterpret_tensor(buf14, (128, 1), (1, 128), 0); del buf14  # reuse
    buf433 = empty((), device='cpu', dtype=torch.float32)
    buf434 = empty((), device='cpu', dtype=torch.int64)
    buf435 = buf433; del buf433  # reuse
    cpp_fused__log_softmax_nll_loss_forward_129(c_void_p(buf435.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(arg191_1.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(buf434.data_ptr()))
    del arg191_1
    return (buf435, reinterpret_tensor(buf430, (1, 128, 250112), (32014336, 250112, 1), 0), reinterpret_tensor(buf3, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf9, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf175, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf180, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf195, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf201, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf209, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf214, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf228, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf234, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf243, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf248, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf262, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf268, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf276, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf281, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf296, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf302, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf310, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf315, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf330, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf336, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf344, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf349, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf363, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf369, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf378, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf383, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf397, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf403, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf411, (1, 6, 128, 64), (49152, 64, 384, 1), 0), reinterpret_tensor(buf416, (1, 6, 128, 64), (49152, 64, 384, 1), 0), buf174, )


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
    arg32_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((250112, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((32, 6), (6, 1), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((32, 6), (6, 1), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((384, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((250112, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    arg191_1 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    arg192_1 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('MT5ForConditionalGeneration', benchmark_compiled_module)
