
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


cpp_fused_add_embedding_native_layer_norm_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const long* in_ptr0,
                       const float* in_ptr1,
                       const long* in_ptr2,
                       const float* in_ptr3,
                       const long* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp5 = in_ptr2[static_cast<long>(x0)];
                        auto tmp11 = in_ptr4[static_cast<long>(x0)];
                        auto tmp1 = decltype(tmp0)(tmp0 + 30522);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 30522L), "index out of bounds: 0 <= tmp3 < 30522L")
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*tmp3)));
                        auto tmp6 = decltype(tmp5)(tmp5 + 512);
                        auto tmp7 = tmp5 < 0;
                        auto tmp8 = tmp7 ? tmp6 : tmp5;
                        TORCH_CHECK((0 <= tmp8) & (tmp8 < 512L), "index out of bounds: 0 <= tmp8 < 512L")
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*tmp8)));
                        auto tmp10 = tmp4 + tmp9;
                        auto tmp12 = decltype(tmp11)(tmp11 + 2);
                        auto tmp13 = tmp11 < 0;
                        auto tmp14 = tmp13 ? tmp12 : tmp11;
                        TORCH_CHECK((0 <= tmp14) & (tmp14 < 2L), "index out of bounds: 0 <= tmp14 < 2L")
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (768L*tmp14)));
                        auto tmp16 = tmp10 + tmp15;
                        tmp16.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = out_ptr2[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-12);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    tmp15.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (768L*x1)), static_cast<long>(768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_view_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp4[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x0 + (384L*x1)), static_cast<long>(384L), tmp4, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                        auto tmp1 = in_ptr1[static_cast<long>(x0 + x0_inner)];
                        auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x0_inner));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 + tmp2;
                        auto tmp6 = tmp3 * tmp5;
                        tmp6.store(out_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 + tmp1;
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                }
                #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp6 = tmp5.exp();
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
}
''')


cpp_fused__softmax_clone_im2col_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       long* out_ptr0,
                       long* out_ptr1,
                       float* out_ptr2)
{
    {
        auto tmp0 = static_cast<long>(0);
        out_ptr0[static_cast<long>(0L)] = tmp0;
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(9L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = c10::convert<long>(x0 + x1);
                out_ptr1[static_cast<long>(x1 + (512L*x0))] = tmp0;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = tmp0 / tmp1;
                in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp2;
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr0[static_cast<long>(0L)];
                        auto tmp1 = decltype(tmp0)(tmp0 + 1);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 1L), "index out of bounds: 0 <= tmp3 < 1L")
                        auto tmp4 = c10::convert<long>((-4L) + x0 + x2);
                        auto tmp5 = static_cast<long>(0);
                        auto tmp6 = tmp4 >= tmp5;
                        auto tmp7 = static_cast<long>(512);
                        auto tmp8 = tmp4 < tmp7;
                        auto tmp9 = tmp6 & tmp8;
                        auto tmp10 = [&]
                        {
                            auto tmp11 = in_ptr1[static_cast<long>((-1536L) + x1 + (384L*x0) + (384L*x2))];
                            return tmp11;
                        }
                        ;
                        auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                        out_ptr2[static_cast<long>(x2 + (9L*x1) + (3456L*x0))] = tmp12;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_5 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
                        auto tmp1 = static_cast<float>(0.3535533905932738);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x0 + (384L*x1)), static_cast<long>(384L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        auto tmp2 = static_cast<float>(0.3535533905932738);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        tmp4.store(out_ptr1 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_6 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const bool* in_ptr0,
                       const float* in_ptr1,
                       bool* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(262144L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (262144L*x0))];
                    out_ptr0[static_cast<long>(x0 + (6L*x1))] = tmp0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(262144L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (262144L*x0))];
                    out_ptr0[static_cast<long>(x0 + (6L*x1))] = tmp0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (32768L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (32768L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(c10::div_floor_integer(x1, 64L));
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(6);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr2[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer(x1, 64L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(12);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr3[static_cast<long>((-384L) + x1 + (384L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    out_ptr2[static_cast<long>(x1 + (768L*x0))] = tmp14;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-12);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_10 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_transpose_view_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (768L*x1)), static_cast<long>(768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_view_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp4[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x0 + (384L*x1)), static_cast<long>(384L), tmp4, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                        auto tmp1 = in_ptr1[static_cast<long>(x0 + x0_inner)];
                        auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x0_inner));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 + tmp2;
                        auto tmp6 = tmp3 * tmp5;
                        tmp6.store(out_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 + tmp1;
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                }
                #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp6 = tmp5.exp();
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
}
''')


cpp_fused__softmax_clone_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = tmp0 / tmp1;
                in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp2;
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(0L)];
                        auto tmp1 = decltype(tmp0)(tmp0 + 1);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 1L), "index out of bounds: 0 <= tmp3 < 1L")
                        auto tmp4 = c10::convert<long>((-4L) + x0 + x2);
                        auto tmp5 = static_cast<long>(0);
                        auto tmp6 = tmp4 >= tmp5;
                        auto tmp7 = static_cast<long>(512);
                        auto tmp8 = tmp4 < tmp7;
                        auto tmp9 = tmp6 & tmp8;
                        auto tmp10 = [&]
                        {
                            auto tmp11 = in_ptr2[static_cast<long>((-1536L) + x1 + (384L*x0) + (384L*x2))];
                            return tmp11;
                        }
                        ;
                        auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                        out_ptr0[static_cast<long>(x2 + (9L*x1) + (3456L*x0))] = tmp12;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_16 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
                        auto tmp1 = static_cast<float>(0.3535533905932738);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x0 + (384L*x1)), static_cast<long>(384L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        auto tmp2 = static_cast<float>(0.3535533905932738);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        tmp4.store(out_ptr1 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_17 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const bool* in_ptr0,
                       const float* in_ptr1,
                       bool* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(262144L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (262144L*x0))];
                    out_ptr0[static_cast<long>(x0 + (6L*x1))] = tmp0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(262144L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (262144L*x0))];
                    out_ptr0[static_cast<long>(x0 + (6L*x1))] = tmp0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (32768L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (32768L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(c10::div_floor_integer(x1, 64L));
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(6);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr2[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer(x1, 64L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(12);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr3[static_cast<long>((-384L) + x1 + (384L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    out_ptr2[static_cast<long>(x1 + (768L*x0))] = tmp14;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_21 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_transpose_view_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (768L*x1)), static_cast<long>(768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_view_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp4[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x0 + (384L*x1)), static_cast<long>(384L), tmp4, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                        auto tmp1 = in_ptr1[static_cast<long>(x0 + x0_inner)];
                        auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x0_inner));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 + tmp2;
                        auto tmp6 = tmp3 * tmp5;
                        tmp6.store(out_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 + tmp1;
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                }
                #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp6 = tmp5.exp();
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
}
''')


cpp_fused__softmax_clone_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = tmp0 / tmp1;
                in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp2;
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(0L)];
                        auto tmp1 = decltype(tmp0)(tmp0 + 1);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 1L), "index out of bounds: 0 <= tmp3 < 1L")
                        auto tmp4 = c10::convert<long>((-4L) + x0 + x2);
                        auto tmp5 = static_cast<long>(0);
                        auto tmp6 = tmp4 >= tmp5;
                        auto tmp7 = static_cast<long>(512);
                        auto tmp8 = tmp4 < tmp7;
                        auto tmp9 = tmp6 & tmp8;
                        auto tmp10 = [&]
                        {
                            auto tmp11 = in_ptr2[static_cast<long>((-1536L) + x1 + (384L*x0) + (384L*x2))];
                            return tmp11;
                        }
                        ;
                        auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                        out_ptr0[static_cast<long>(x2 + (9L*x1) + (3456L*x0))] = tmp12;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_27 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
                        auto tmp1 = static_cast<float>(0.3535533905932738);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x0 + (384L*x1)), static_cast<long>(384L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        auto tmp2 = static_cast<float>(0.3535533905932738);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        tmp4.store(out_ptr1 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_28 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const bool* in_ptr0,
                       const float* in_ptr1,
                       bool* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(262144L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (262144L*x0))];
                    out_ptr0[static_cast<long>(x0 + (6L*x1))] = tmp0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(262144L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (262144L*x0))];
                    out_ptr0[static_cast<long>(x0 + (6L*x1))] = tmp0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (32768L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (32768L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(c10::div_floor_integer(x1, 64L));
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(6);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr2[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer(x1, 64L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(12);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr3[static_cast<long>((-384L) + x1 + (384L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    out_ptr2[static_cast<long>(x1 + (768L*x0))] = tmp14;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_32 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_transpose_view_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (768L*x1)), static_cast<long>(768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_view_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp4[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x0 + (384L*x1)), static_cast<long>(384L), tmp4, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                        auto tmp1 = in_ptr1[static_cast<long>(x0 + x0_inner)];
                        auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x0_inner));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 + tmp2;
                        auto tmp6 = tmp3 * tmp5;
                        tmp6.store(out_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 + tmp1;
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                }
                #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp6 = tmp5.exp();
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
}
''')


cpp_fused__softmax_clone_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = tmp0 / tmp1;
                in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp2;
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(0L)];
                        auto tmp1 = decltype(tmp0)(tmp0 + 1);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 1L), "index out of bounds: 0 <= tmp3 < 1L")
                        auto tmp4 = c10::convert<long>((-4L) + x0 + x2);
                        auto tmp5 = static_cast<long>(0);
                        auto tmp6 = tmp4 >= tmp5;
                        auto tmp7 = static_cast<long>(512);
                        auto tmp8 = tmp4 < tmp7;
                        auto tmp9 = tmp6 & tmp8;
                        auto tmp10 = [&]
                        {
                            auto tmp11 = in_ptr2[static_cast<long>((-1536L) + x1 + (384L*x0) + (384L*x2))];
                            return tmp11;
                        }
                        ;
                        auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                        out_ptr0[static_cast<long>(x2 + (9L*x1) + (3456L*x0))] = tmp12;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_38 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
                        auto tmp1 = static_cast<float>(0.3535533905932738);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x0 + (384L*x1)), static_cast<long>(384L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        auto tmp2 = static_cast<float>(0.3535533905932738);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        tmp4.store(out_ptr1 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_39 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const bool* in_ptr0,
                       const float* in_ptr1,
                       bool* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(262144L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (262144L*x0))];
                    out_ptr0[static_cast<long>(x0 + (6L*x1))] = tmp0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(262144L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (262144L*x0))];
                    out_ptr0[static_cast<long>(x0 + (6L*x1))] = tmp0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (32768L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (32768L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(c10::div_floor_integer(x1, 64L));
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(6);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr2[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer(x1, 64L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(12);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr3[static_cast<long>((-384L) + x1 + (384L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    out_ptr2[static_cast<long>(x1 + (768L*x0))] = tmp14;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_43 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_transpose_view_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (768L*x1)), static_cast<long>(768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_view_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp4[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x0 + (384L*x1)), static_cast<long>(384L), tmp4, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                        auto tmp1 = in_ptr1[static_cast<long>(x0 + x0_inner)];
                        auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x0_inner));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 + tmp2;
                        auto tmp6 = tmp3 * tmp5;
                        tmp6.store(out_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 + tmp1;
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                }
                #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp6 = tmp5.exp();
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
}
''')


cpp_fused__softmax_clone_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = tmp0 / tmp1;
                in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp2;
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(0L)];
                        auto tmp1 = decltype(tmp0)(tmp0 + 1);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 1L), "index out of bounds: 0 <= tmp3 < 1L")
                        auto tmp4 = c10::convert<long>((-4L) + x0 + x2);
                        auto tmp5 = static_cast<long>(0);
                        auto tmp6 = tmp4 >= tmp5;
                        auto tmp7 = static_cast<long>(512);
                        auto tmp8 = tmp4 < tmp7;
                        auto tmp9 = tmp6 & tmp8;
                        auto tmp10 = [&]
                        {
                            auto tmp11 = in_ptr2[static_cast<long>((-1536L) + x1 + (384L*x0) + (384L*x2))];
                            return tmp11;
                        }
                        ;
                        auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                        out_ptr0[static_cast<long>(x2 + (9L*x1) + (3456L*x0))] = tmp12;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_49 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
                        auto tmp1 = static_cast<float>(0.3535533905932738);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x0 + (384L*x1)), static_cast<long>(384L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        auto tmp2 = static_cast<float>(0.3535533905932738);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        tmp4.store(out_ptr1 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_50 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const bool* in_ptr0,
                       const float* in_ptr1,
                       bool* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(262144L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (262144L*x0))];
                    out_ptr0[static_cast<long>(x0 + (6L*x1))] = tmp0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
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
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(262144L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (262144L*x0))];
                    out_ptr0[static_cast<long>(x0 + (6L*x1))] = tmp0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (32768L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (32768L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(c10::div_floor_integer(x1, 64L));
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(6);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr2[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer(x1, 64L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(12);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr3[static_cast<long>((-384L) + x1 + (384L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    out_ptr2[static_cast<long>(x1 + (768L*x0))] = tmp14;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_54 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_transpose_view_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (768L*x1)), static_cast<long>(768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_view_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp4[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x0 + (384L*x1)), static_cast<long>(384L), tmp4, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                        auto tmp1 = in_ptr1[static_cast<long>(x0 + x0_inner)];
                        auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x0_inner));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 + tmp2;
                        auto tmp6 = tmp3 * tmp5;
                        tmp6.store(out_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 + tmp1;
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                }
                #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp6 = tmp5.exp();
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
}
''')


cpp_fused__softmax_clone_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = tmp0 / tmp1;
                in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp2;
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(0L)];
                        auto tmp1 = decltype(tmp0)(tmp0 + 1);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 1L), "index out of bounds: 0 <= tmp3 < 1L")
                        auto tmp4 = c10::convert<long>((-4L) + x0 + x2);
                        auto tmp5 = static_cast<long>(0);
                        auto tmp6 = tmp4 >= tmp5;
                        auto tmp7 = static_cast<long>(512);
                        auto tmp8 = tmp4 < tmp7;
                        auto tmp9 = tmp6 & tmp8;
                        auto tmp10 = [&]
                        {
                            auto tmp11 = in_ptr2[static_cast<long>((-1536L) + x1 + (384L*x0) + (384L*x2))];
                            return tmp11;
                        }
                        ;
                        auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                        out_ptr0[static_cast<long>(x2 + (9L*x1) + (3456L*x0))] = tmp12;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_60 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
                        auto tmp1 = static_cast<float>(0.3535533905932738);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x0 + (384L*x1)), static_cast<long>(384L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        auto tmp2 = static_cast<float>(0.3535533905932738);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        tmp4.store(out_ptr1 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_61 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const bool* in_ptr0,
                       const float* in_ptr1,
                       bool* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(262144L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (262144L*x0))];
                    out_ptr0[static_cast<long>(x0 + (6L*x1))] = tmp0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
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
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(262144L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (262144L*x0))];
                    out_ptr0[static_cast<long>(x0 + (6L*x1))] = tmp0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (32768L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (32768L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(c10::div_floor_integer(x1, 64L));
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(6);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr2[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer(x1, 64L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(12);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr3[static_cast<long>((-384L) + x1 + (384L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    out_ptr2[static_cast<long>(x1 + (768L*x0))] = tmp14;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_65 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_transpose_view_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (768L*x1)), static_cast<long>(768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_view_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp4[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x0 + (384L*x1)), static_cast<long>(384L), tmp4, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                        auto tmp1 = in_ptr1[static_cast<long>(x0 + x0_inner)];
                        auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x0_inner));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 + tmp2;
                        auto tmp6 = tmp3 * tmp5;
                        tmp6.store(out_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 + tmp1;
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                }
                #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp6 = tmp5.exp();
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
}
''')


cpp_fused__softmax_clone_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = tmp0 / tmp1;
                in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp2;
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(0L)];
                        auto tmp1 = decltype(tmp0)(tmp0 + 1);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 1L), "index out of bounds: 0 <= tmp3 < 1L")
                        auto tmp4 = c10::convert<long>((-4L) + x0 + x2);
                        auto tmp5 = static_cast<long>(0);
                        auto tmp6 = tmp4 >= tmp5;
                        auto tmp7 = static_cast<long>(512);
                        auto tmp8 = tmp4 < tmp7;
                        auto tmp9 = tmp6 & tmp8;
                        auto tmp10 = [&]
                        {
                            auto tmp11 = in_ptr2[static_cast<long>((-1536L) + x1 + (384L*x0) + (384L*x2))];
                            return tmp11;
                        }
                        ;
                        auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                        out_ptr0[static_cast<long>(x2 + (9L*x1) + (3456L*x0))] = tmp12;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_71 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
                        auto tmp1 = static_cast<float>(0.3535533905932738);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x0 + (384L*x1)), static_cast<long>(384L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        auto tmp2 = static_cast<float>(0.3535533905932738);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        tmp4.store(out_ptr1 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_72 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const bool* in_ptr0,
                       const float* in_ptr1,
                       bool* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(262144L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (262144L*x0))];
                    out_ptr0[static_cast<long>(x0 + (6L*x1))] = tmp0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(262144L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (262144L*x0))];
                    out_ptr0[static_cast<long>(x0 + (6L*x1))] = tmp0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (32768L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (32768L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(c10::div_floor_integer(x1, 64L));
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(6);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr2[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer(x1, 64L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(12);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr3[static_cast<long>((-384L) + x1 + (384L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    out_ptr2[static_cast<long>(x1 + (768L*x0))] = tmp14;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_76 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_transpose_view_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (768L*x1)), static_cast<long>(768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_view_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp4[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x0 + (384L*x1)), static_cast<long>(384L), tmp4, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                        auto tmp1 = in_ptr1[static_cast<long>(x0 + x0_inner)];
                        auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x0_inner));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 + tmp2;
                        auto tmp6 = tmp3 * tmp5;
                        tmp6.store(out_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 + tmp1;
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                }
                #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp6 = tmp5.exp();
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
}
''')


cpp_fused__softmax_clone_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = tmp0 / tmp1;
                in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp2;
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(0L)];
                        auto tmp1 = decltype(tmp0)(tmp0 + 1);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 1L), "index out of bounds: 0 <= tmp3 < 1L")
                        auto tmp4 = c10::convert<long>((-4L) + x0 + x2);
                        auto tmp5 = static_cast<long>(0);
                        auto tmp6 = tmp4 >= tmp5;
                        auto tmp7 = static_cast<long>(512);
                        auto tmp8 = tmp4 < tmp7;
                        auto tmp9 = tmp6 & tmp8;
                        auto tmp10 = [&]
                        {
                            auto tmp11 = in_ptr2[static_cast<long>((-1536L) + x1 + (384L*x0) + (384L*x2))];
                            return tmp11;
                        }
                        ;
                        auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                        out_ptr0[static_cast<long>(x2 + (9L*x1) + (3456L*x0))] = tmp12;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_82 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
                        auto tmp1 = static_cast<float>(0.3535533905932738);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x0 + (384L*x1)), static_cast<long>(384L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        auto tmp2 = static_cast<float>(0.3535533905932738);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        tmp4.store(out_ptr1 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_83 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const bool* in_ptr0,
                       const float* in_ptr1,
                       bool* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(262144L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (262144L*x0))];
                    out_ptr0[static_cast<long>(x0 + (6L*x1))] = tmp0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_85 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(262144L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (262144L*x0))];
                    out_ptr0[static_cast<long>(x0 + (6L*x1))] = tmp0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (32768L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (32768L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(c10::div_floor_integer(x1, 64L));
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(6);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr2[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer(x1, 64L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(12);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr3[static_cast<long>((-384L) + x1 + (384L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    out_ptr2[static_cast<long>(x1 + (768L*x0))] = tmp14;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_86 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_87 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_transpose_view_88 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (768L*x1)), static_cast<long>(768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_view_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp4[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x0 + (384L*x1)), static_cast<long>(384L), tmp4, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                        auto tmp1 = in_ptr1[static_cast<long>(x0 + x0_inner)];
                        auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x0_inner));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 + tmp2;
                        auto tmp6 = tmp3 * tmp5;
                        tmp6.store(out_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_91 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 + tmp1;
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                }
                #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp6 = tmp5.exp();
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
}
''')


cpp_fused__softmax_clone_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = tmp0 / tmp1;
                in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp2;
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(0L)];
                        auto tmp1 = decltype(tmp0)(tmp0 + 1);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 1L), "index out of bounds: 0 <= tmp3 < 1L")
                        auto tmp4 = c10::convert<long>((-4L) + x0 + x2);
                        auto tmp5 = static_cast<long>(0);
                        auto tmp6 = tmp4 >= tmp5;
                        auto tmp7 = static_cast<long>(512);
                        auto tmp8 = tmp4 < tmp7;
                        auto tmp9 = tmp6 & tmp8;
                        auto tmp10 = [&]
                        {
                            auto tmp11 = in_ptr2[static_cast<long>((-1536L) + x1 + (384L*x0) + (384L*x2))];
                            return tmp11;
                        }
                        ;
                        auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                        out_ptr0[static_cast<long>(x2 + (9L*x1) + (3456L*x0))] = tmp12;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_93 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
                        auto tmp1 = static_cast<float>(0.3535533905932738);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x0 + (384L*x1)), static_cast<long>(384L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        auto tmp2 = static_cast<float>(0.3535533905932738);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        tmp4.store(out_ptr1 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_94 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_95 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const bool* in_ptr0,
                       const float* in_ptr1,
                       bool* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(262144L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (262144L*x0))];
                    out_ptr0[static_cast<long>(x0 + (6L*x1))] = tmp0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_96 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(262144L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (262144L*x0))];
                    out_ptr0[static_cast<long>(x0 + (6L*x1))] = tmp0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (32768L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (32768L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(c10::div_floor_integer(x1, 64L));
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(6);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr2[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer(x1, 64L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(12);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr3[static_cast<long>((-384L) + x1 + (384L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    out_ptr2[static_cast<long>(x1 + (768L*x0))] = tmp14;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_97 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_98 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_transpose_view_99 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_100 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (768L*x1)), static_cast<long>(768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_view_101 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp4[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x0 + (384L*x1)), static_cast<long>(384L), tmp4, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                        auto tmp1 = in_ptr1[static_cast<long>(x0 + x0_inner)];
                        auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x0_inner));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 + tmp2;
                        auto tmp6 = tmp3 * tmp5;
                        tmp6.store(out_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_102 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 + tmp1;
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                }
                #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp6 = tmp5.exp();
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
}
''')


cpp_fused__softmax_clone_103 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = tmp0 / tmp1;
                in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp2;
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(0L)];
                        auto tmp1 = decltype(tmp0)(tmp0 + 1);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 1L), "index out of bounds: 0 <= tmp3 < 1L")
                        auto tmp4 = c10::convert<long>((-4L) + x0 + x2);
                        auto tmp5 = static_cast<long>(0);
                        auto tmp6 = tmp4 >= tmp5;
                        auto tmp7 = static_cast<long>(512);
                        auto tmp8 = tmp4 < tmp7;
                        auto tmp9 = tmp6 & tmp8;
                        auto tmp10 = [&]
                        {
                            auto tmp11 = in_ptr2[static_cast<long>((-1536L) + x1 + (384L*x0) + (384L*x2))];
                            return tmp11;
                        }
                        ;
                        auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                        out_ptr0[static_cast<long>(x2 + (9L*x1) + (3456L*x0))] = tmp12;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_104 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
                        auto tmp1 = static_cast<float>(0.3535533905932738);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x0 + (384L*x1)), static_cast<long>(384L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        auto tmp2 = static_cast<float>(0.3535533905932738);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        tmp4.store(out_ptr1 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_105 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_106 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const bool* in_ptr0,
                       const float* in_ptr1,
                       bool* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(262144L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (262144L*x0))];
                    out_ptr0[static_cast<long>(x0 + (6L*x1))] = tmp0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_107 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(262144L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (262144L*x0))];
                    out_ptr0[static_cast<long>(x0 + (6L*x1))] = tmp0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (32768L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (32768L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(c10::div_floor_integer(x1, 64L));
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(6);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr2[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer(x1, 64L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(12);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr3[static_cast<long>((-384L) + x1 + (384L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    out_ptr2[static_cast<long>(x1 + (768L*x0))] = tmp14;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_108 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_109 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_transpose_view_110 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_111 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (768L*x1)), static_cast<long>(768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_view_112 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp4[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x0 + (384L*x1)), static_cast<long>(384L), tmp4, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                        auto tmp1 = in_ptr1[static_cast<long>(x0 + x0_inner)];
                        auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x0_inner));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 + tmp2;
                        auto tmp6 = tmp3 * tmp5;
                        tmp6.store(out_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_113 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 + tmp1;
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                }
                #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp6 = tmp5.exp();
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
}
''')


cpp_fused__softmax_clone_114 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = tmp0 / tmp1;
                in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp2;
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(0L)];
                        auto tmp1 = decltype(tmp0)(tmp0 + 1);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 1L), "index out of bounds: 0 <= tmp3 < 1L")
                        auto tmp4 = c10::convert<long>((-4L) + x0 + x2);
                        auto tmp5 = static_cast<long>(0);
                        auto tmp6 = tmp4 >= tmp5;
                        auto tmp7 = static_cast<long>(512);
                        auto tmp8 = tmp4 < tmp7;
                        auto tmp9 = tmp6 & tmp8;
                        auto tmp10 = [&]
                        {
                            auto tmp11 = in_ptr2[static_cast<long>((-1536L) + x1 + (384L*x0) + (384L*x2))];
                            return tmp11;
                        }
                        ;
                        auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                        out_ptr0[static_cast<long>(x2 + (9L*x1) + (3456L*x0))] = tmp12;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_115 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
                        auto tmp1 = static_cast<float>(0.3535533905932738);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x0 + (384L*x1)), static_cast<long>(384L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        auto tmp2 = static_cast<float>(0.3535533905932738);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        tmp4.store(out_ptr1 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_116 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_117 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const bool* in_ptr0,
                       const float* in_ptr1,
                       bool* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(262144L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (262144L*x0))];
                    out_ptr0[static_cast<long>(x0 + (6L*x1))] = tmp0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_118 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(262144L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (262144L*x0))];
                    out_ptr0[static_cast<long>(x0 + (6L*x1))] = tmp0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (32768L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (32768L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(c10::div_floor_integer(x1, 64L));
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(6);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr2[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer(x1, 64L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(12);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr3[static_cast<long>((-384L) + x1 + (384L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    out_ptr2[static_cast<long>(x1 + (768L*x0))] = tmp14;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_119 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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


cpp_fused_add_native_layer_norm_transpose_view_121 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_122 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x0 + (768L*x1)), static_cast<long>(768L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        tmp1.store(out_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mul_view_123 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp4[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x0 + (384L*x1)), static_cast<long>(384L), tmp4, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                        auto tmp1 = in_ptr1[static_cast<long>(x0 + x0_inner)];
                        auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x0_inner));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 + tmp2;
                        auto tmp6 = tmp3 * tmp5;
                        tmp6.store(out_ptr0 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_124 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = tmp0 + tmp1;
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                }
                #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(static_cast<long>((x1 + x1_inner + (9L*x0))) % static_cast<long>(54L))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp6 = tmp5.exp();
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp6;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(static_cast<long>((x1 + (9L*x0))) % static_cast<long>(54L))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
                    tmp_acc0 = tmp_acc0 + tmp5;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
}
''')


cpp_fused__softmax_clone_125 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp2 = tmp0 / tmp1;
                in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp2;
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(0L)];
                        auto tmp1 = decltype(tmp0)(tmp0 + 1);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 1L), "index out of bounds: 0 <= tmp3 < 1L")
                        auto tmp4 = c10::convert<long>((-4L) + x0 + x2);
                        auto tmp5 = static_cast<long>(0);
                        auto tmp6 = tmp4 >= tmp5;
                        auto tmp7 = static_cast<long>(512);
                        auto tmp8 = tmp4 < tmp7;
                        auto tmp9 = tmp6 & tmp8;
                        auto tmp10 = [&]
                        {
                            auto tmp11 = in_ptr2[static_cast<long>((-1536L) + x1 + (384L*x0) + (384L*x2))];
                            return tmp11;
                        }
                        ;
                        auto tmp12 = tmp9 ? tmp10() : static_cast<decltype(tmp10())>(0.0);
                        out_ptr0[static_cast<long>(x2 + (9L*x1) + (3456L*x0))] = tmp12;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_126 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
                        auto tmp1 = static_cast<float>(0.3535533905932738);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x0 + (384L*x1)), static_cast<long>(384L), tmp0, 8);
                    for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x0_inner));
                        auto tmp2 = static_cast<float>(0.3535533905932738);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 * tmp3;
                        tmp4.store(out_ptr1 + static_cast<long>(x1 + (512L*x0) + (512L*x0_inner)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_127 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_128 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const bool* in_ptr0,
                       const float* in_ptr1,
                       bool* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(262144L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (262144L*x0))];
                    out_ptr0[static_cast<long>(x0 + (6L*x1))] = tmp0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (384L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (32768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_129 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(262144L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (262144L*x0))];
                    out_ptr0[static_cast<long>(x0 + (6L*x1))] = tmp0;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (32768L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (32768L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(c10::div_floor_integer(x1, 64L));
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(6);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr2[static_cast<long>((64L*x0) + (32768L*(c10::div_floor_integer(x1, 64L))) + (static_cast<long>(x1) % static_cast<long>(64L)))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(12);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr3[static_cast<long>((-384L) + x1 + (384L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    out_ptr2[static_cast<long>(x1 + (768L*x0))] = tmp14;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_130 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_131 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_view_132 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-12);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_native_layer_norm_view_133 = async_compile.cpp('''
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
                    auto tmp18 = static_cast<float>(1e-12);
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


cpp_fused__log_softmax_add_gelu_native_layer_norm_native_layer_norm_backward_nll_loss_forward_134 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(30520L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (30522L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(30520L); x1<static_cast<long>(30522L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (30522L*x0))];
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(30520L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (30522L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(30520L); x1<static_cast<long>(30522L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (30522L*x0))];
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(30520L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (30522L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = std::log(tmp4);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp3 - tmp6;
                    tmp7.store(out_ptr2 + static_cast<long>(x1 + (30522L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(30520L); x1<static_cast<long>(30522L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (30522L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = std::log(tmp3);
                    auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                    out_ptr2[static_cast<long>(x1 + (30522L*x0))] = tmp5;
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
                        auto tmp6 = decltype(tmp5)(tmp5 + 30522);
                        auto tmp7 = tmp5 < 0;
                        auto tmp8 = tmp7 ? tmp6 : tmp5;
                        TORCH_CHECK((0 <= tmp8) & (tmp8 < 30522L), "index out of bounds: 0 <= tmp8 < 30522L")
                        auto tmp9 = out_ptr2[static_cast<long>(tmp8 + (30522L*x0))];
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
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
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
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr5 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
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
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr6 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr7 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr7 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr8 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
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
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr9 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr10 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr10 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr11 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
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
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr12 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr13 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr13 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr14 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
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
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr15 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr16 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr16 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr17 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
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
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr18 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr19 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr19 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr20 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
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
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr21 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr22 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr22 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr23 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
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
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr24 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr25 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr25 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr26 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr26 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291 = args
    args.clear()
    assert_size_stride(primals_1, (384, 1), (1, 1))
    assert_size_stride(primals_2, (384, 1), (1, 1))
    assert_size_stride(primals_3, (384, 1), (1, 1))
    assert_size_stride(primals_4, (384, 1), (1, 1))
    assert_size_stride(primals_5, (384, 1), (1, 1))
    assert_size_stride(primals_6, (384, 1), (1, 1))
    assert_size_stride(primals_7, (384, 1), (1, 1))
    assert_size_stride(primals_8, (384, 1), (1, 1))
    assert_size_stride(primals_9, (384, 1), (1, 1))
    assert_size_stride(primals_10, (384, 1), (1, 1))
    assert_size_stride(primals_11, (384, 1), (1, 1))
    assert_size_stride(primals_12, (384, 1), (1, 1))
    assert_size_stride(primals_13, (30522, 768), (768, 1))
    assert_size_stride(primals_14, (512, 768), (768, 1))
    assert_size_stride(primals_15, (2, 768), (768, 1))
    assert_size_stride(primals_16, (768, ), (1, ))
    assert_size_stride(primals_17, (768, ), (1, ))
    assert_size_stride(primals_18, (384, 768), (768, 1))
    assert_size_stride(primals_19, (384, ), (1, ))
    assert_size_stride(primals_20, (384, 768), (768, 1))
    assert_size_stride(primals_21, (384, ), (1, ))
    assert_size_stride(primals_22, (384, 768), (768, 1))
    assert_size_stride(primals_23, (384, ), (1, ))
    assert_size_stride(primals_24, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_25, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_26, (54, 384), (384, 1))
    assert_size_stride(primals_27, (54, ), (1, ))
    assert_size_stride(primals_28, (384, 768), (768, 1))
    assert_size_stride(primals_29, (384, ), (1, ))
    assert_size_stride(primals_30, (768, 768), (768, 1))
    assert_size_stride(primals_31, (768, ), (1, ))
    assert_size_stride(primals_32, (768, ), (1, ))
    assert_size_stride(primals_33, (768, ), (1, ))
    assert_size_stride(primals_34, (3072, 768), (768, 1))
    assert_size_stride(primals_35, (3072, ), (1, ))
    assert_size_stride(primals_36, (768, 3072), (3072, 1))
    assert_size_stride(primals_37, (768, ), (1, ))
    assert_size_stride(primals_38, (768, ), (1, ))
    assert_size_stride(primals_39, (768, ), (1, ))
    assert_size_stride(primals_40, (384, 768), (768, 1))
    assert_size_stride(primals_41, (384, ), (1, ))
    assert_size_stride(primals_42, (384, 768), (768, 1))
    assert_size_stride(primals_43, (384, ), (1, ))
    assert_size_stride(primals_44, (384, 768), (768, 1))
    assert_size_stride(primals_45, (384, ), (1, ))
    assert_size_stride(primals_46, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_47, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_48, (54, 384), (384, 1))
    assert_size_stride(primals_49, (54, ), (1, ))
    assert_size_stride(primals_50, (384, 768), (768, 1))
    assert_size_stride(primals_51, (384, ), (1, ))
    assert_size_stride(primals_52, (768, 768), (768, 1))
    assert_size_stride(primals_53, (768, ), (1, ))
    assert_size_stride(primals_54, (768, ), (1, ))
    assert_size_stride(primals_55, (768, ), (1, ))
    assert_size_stride(primals_56, (3072, 768), (768, 1))
    assert_size_stride(primals_57, (3072, ), (1, ))
    assert_size_stride(primals_58, (768, 3072), (3072, 1))
    assert_size_stride(primals_59, (768, ), (1, ))
    assert_size_stride(primals_60, (768, ), (1, ))
    assert_size_stride(primals_61, (768, ), (1, ))
    assert_size_stride(primals_62, (384, 768), (768, 1))
    assert_size_stride(primals_63, (384, ), (1, ))
    assert_size_stride(primals_64, (384, 768), (768, 1))
    assert_size_stride(primals_65, (384, ), (1, ))
    assert_size_stride(primals_66, (384, 768), (768, 1))
    assert_size_stride(primals_67, (384, ), (1, ))
    assert_size_stride(primals_68, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_69, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_70, (54, 384), (384, 1))
    assert_size_stride(primals_71, (54, ), (1, ))
    assert_size_stride(primals_72, (384, 768), (768, 1))
    assert_size_stride(primals_73, (384, ), (1, ))
    assert_size_stride(primals_74, (768, 768), (768, 1))
    assert_size_stride(primals_75, (768, ), (1, ))
    assert_size_stride(primals_76, (768, ), (1, ))
    assert_size_stride(primals_77, (768, ), (1, ))
    assert_size_stride(primals_78, (3072, 768), (768, 1))
    assert_size_stride(primals_79, (3072, ), (1, ))
    assert_size_stride(primals_80, (768, 3072), (3072, 1))
    assert_size_stride(primals_81, (768, ), (1, ))
    assert_size_stride(primals_82, (768, ), (1, ))
    assert_size_stride(primals_83, (768, ), (1, ))
    assert_size_stride(primals_84, (384, 768), (768, 1))
    assert_size_stride(primals_85, (384, ), (1, ))
    assert_size_stride(primals_86, (384, 768), (768, 1))
    assert_size_stride(primals_87, (384, ), (1, ))
    assert_size_stride(primals_88, (384, 768), (768, 1))
    assert_size_stride(primals_89, (384, ), (1, ))
    assert_size_stride(primals_90, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_91, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_92, (54, 384), (384, 1))
    assert_size_stride(primals_93, (54, ), (1, ))
    assert_size_stride(primals_94, (384, 768), (768, 1))
    assert_size_stride(primals_95, (384, ), (1, ))
    assert_size_stride(primals_96, (768, 768), (768, 1))
    assert_size_stride(primals_97, (768, ), (1, ))
    assert_size_stride(primals_98, (768, ), (1, ))
    assert_size_stride(primals_99, (768, ), (1, ))
    assert_size_stride(primals_100, (3072, 768), (768, 1))
    assert_size_stride(primals_101, (3072, ), (1, ))
    assert_size_stride(primals_102, (768, 3072), (3072, 1))
    assert_size_stride(primals_103, (768, ), (1, ))
    assert_size_stride(primals_104, (768, ), (1, ))
    assert_size_stride(primals_105, (768, ), (1, ))
    assert_size_stride(primals_106, (384, 768), (768, 1))
    assert_size_stride(primals_107, (384, ), (1, ))
    assert_size_stride(primals_108, (384, 768), (768, 1))
    assert_size_stride(primals_109, (384, ), (1, ))
    assert_size_stride(primals_110, (384, 768), (768, 1))
    assert_size_stride(primals_111, (384, ), (1, ))
    assert_size_stride(primals_112, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_113, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_114, (54, 384), (384, 1))
    assert_size_stride(primals_115, (54, ), (1, ))
    assert_size_stride(primals_116, (384, 768), (768, 1))
    assert_size_stride(primals_117, (384, ), (1, ))
    assert_size_stride(primals_118, (768, 768), (768, 1))
    assert_size_stride(primals_119, (768, ), (1, ))
    assert_size_stride(primals_120, (768, ), (1, ))
    assert_size_stride(primals_121, (768, ), (1, ))
    assert_size_stride(primals_122, (3072, 768), (768, 1))
    assert_size_stride(primals_123, (3072, ), (1, ))
    assert_size_stride(primals_124, (768, 3072), (3072, 1))
    assert_size_stride(primals_125, (768, ), (1, ))
    assert_size_stride(primals_126, (768, ), (1, ))
    assert_size_stride(primals_127, (768, ), (1, ))
    assert_size_stride(primals_128, (384, 768), (768, 1))
    assert_size_stride(primals_129, (384, ), (1, ))
    assert_size_stride(primals_130, (384, 768), (768, 1))
    assert_size_stride(primals_131, (384, ), (1, ))
    assert_size_stride(primals_132, (384, 768), (768, 1))
    assert_size_stride(primals_133, (384, ), (1, ))
    assert_size_stride(primals_134, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_135, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_136, (54, 384), (384, 1))
    assert_size_stride(primals_137, (54, ), (1, ))
    assert_size_stride(primals_138, (384, 768), (768, 1))
    assert_size_stride(primals_139, (384, ), (1, ))
    assert_size_stride(primals_140, (768, 768), (768, 1))
    assert_size_stride(primals_141, (768, ), (1, ))
    assert_size_stride(primals_142, (768, ), (1, ))
    assert_size_stride(primals_143, (768, ), (1, ))
    assert_size_stride(primals_144, (3072, 768), (768, 1))
    assert_size_stride(primals_145, (3072, ), (1, ))
    assert_size_stride(primals_146, (768, 3072), (3072, 1))
    assert_size_stride(primals_147, (768, ), (1, ))
    assert_size_stride(primals_148, (768, ), (1, ))
    assert_size_stride(primals_149, (768, ), (1, ))
    assert_size_stride(primals_150, (384, 768), (768, 1))
    assert_size_stride(primals_151, (384, ), (1, ))
    assert_size_stride(primals_152, (384, 768), (768, 1))
    assert_size_stride(primals_153, (384, ), (1, ))
    assert_size_stride(primals_154, (384, 768), (768, 1))
    assert_size_stride(primals_155, (384, ), (1, ))
    assert_size_stride(primals_156, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_157, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_158, (54, 384), (384, 1))
    assert_size_stride(primals_159, (54, ), (1, ))
    assert_size_stride(primals_160, (384, 768), (768, 1))
    assert_size_stride(primals_161, (384, ), (1, ))
    assert_size_stride(primals_162, (768, 768), (768, 1))
    assert_size_stride(primals_163, (768, ), (1, ))
    assert_size_stride(primals_164, (768, ), (1, ))
    assert_size_stride(primals_165, (768, ), (1, ))
    assert_size_stride(primals_166, (3072, 768), (768, 1))
    assert_size_stride(primals_167, (3072, ), (1, ))
    assert_size_stride(primals_168, (768, 3072), (3072, 1))
    assert_size_stride(primals_169, (768, ), (1, ))
    assert_size_stride(primals_170, (768, ), (1, ))
    assert_size_stride(primals_171, (768, ), (1, ))
    assert_size_stride(primals_172, (384, 768), (768, 1))
    assert_size_stride(primals_173, (384, ), (1, ))
    assert_size_stride(primals_174, (384, 768), (768, 1))
    assert_size_stride(primals_175, (384, ), (1, ))
    assert_size_stride(primals_176, (384, 768), (768, 1))
    assert_size_stride(primals_177, (384, ), (1, ))
    assert_size_stride(primals_178, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_179, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_180, (54, 384), (384, 1))
    assert_size_stride(primals_181, (54, ), (1, ))
    assert_size_stride(primals_182, (384, 768), (768, 1))
    assert_size_stride(primals_183, (384, ), (1, ))
    assert_size_stride(primals_184, (768, 768), (768, 1))
    assert_size_stride(primals_185, (768, ), (1, ))
    assert_size_stride(primals_186, (768, ), (1, ))
    assert_size_stride(primals_187, (768, ), (1, ))
    assert_size_stride(primals_188, (3072, 768), (768, 1))
    assert_size_stride(primals_189, (3072, ), (1, ))
    assert_size_stride(primals_190, (768, 3072), (3072, 1))
    assert_size_stride(primals_191, (768, ), (1, ))
    assert_size_stride(primals_192, (768, ), (1, ))
    assert_size_stride(primals_193, (768, ), (1, ))
    assert_size_stride(primals_194, (384, 768), (768, 1))
    assert_size_stride(primals_195, (384, ), (1, ))
    assert_size_stride(primals_196, (384, 768), (768, 1))
    assert_size_stride(primals_197, (384, ), (1, ))
    assert_size_stride(primals_198, (384, 768), (768, 1))
    assert_size_stride(primals_199, (384, ), (1, ))
    assert_size_stride(primals_200, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_201, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_202, (54, 384), (384, 1))
    assert_size_stride(primals_203, (54, ), (1, ))
    assert_size_stride(primals_204, (384, 768), (768, 1))
    assert_size_stride(primals_205, (384, ), (1, ))
    assert_size_stride(primals_206, (768, 768), (768, 1))
    assert_size_stride(primals_207, (768, ), (1, ))
    assert_size_stride(primals_208, (768, ), (1, ))
    assert_size_stride(primals_209, (768, ), (1, ))
    assert_size_stride(primals_210, (3072, 768), (768, 1))
    assert_size_stride(primals_211, (3072, ), (1, ))
    assert_size_stride(primals_212, (768, 3072), (3072, 1))
    assert_size_stride(primals_213, (768, ), (1, ))
    assert_size_stride(primals_214, (768, ), (1, ))
    assert_size_stride(primals_215, (768, ), (1, ))
    assert_size_stride(primals_216, (384, 768), (768, 1))
    assert_size_stride(primals_217, (384, ), (1, ))
    assert_size_stride(primals_218, (384, 768), (768, 1))
    assert_size_stride(primals_219, (384, ), (1, ))
    assert_size_stride(primals_220, (384, 768), (768, 1))
    assert_size_stride(primals_221, (384, ), (1, ))
    assert_size_stride(primals_222, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_223, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_224, (54, 384), (384, 1))
    assert_size_stride(primals_225, (54, ), (1, ))
    assert_size_stride(primals_226, (384, 768), (768, 1))
    assert_size_stride(primals_227, (384, ), (1, ))
    assert_size_stride(primals_228, (768, 768), (768, 1))
    assert_size_stride(primals_229, (768, ), (1, ))
    assert_size_stride(primals_230, (768, ), (1, ))
    assert_size_stride(primals_231, (768, ), (1, ))
    assert_size_stride(primals_232, (3072, 768), (768, 1))
    assert_size_stride(primals_233, (3072, ), (1, ))
    assert_size_stride(primals_234, (768, 3072), (3072, 1))
    assert_size_stride(primals_235, (768, ), (1, ))
    assert_size_stride(primals_236, (768, ), (1, ))
    assert_size_stride(primals_237, (768, ), (1, ))
    assert_size_stride(primals_238, (384, 768), (768, 1))
    assert_size_stride(primals_239, (384, ), (1, ))
    assert_size_stride(primals_240, (384, 768), (768, 1))
    assert_size_stride(primals_241, (384, ), (1, ))
    assert_size_stride(primals_242, (384, 768), (768, 1))
    assert_size_stride(primals_243, (384, ), (1, ))
    assert_size_stride(primals_244, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_245, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_246, (54, 384), (384, 1))
    assert_size_stride(primals_247, (54, ), (1, ))
    assert_size_stride(primals_248, (384, 768), (768, 1))
    assert_size_stride(primals_249, (384, ), (1, ))
    assert_size_stride(primals_250, (768, 768), (768, 1))
    assert_size_stride(primals_251, (768, ), (1, ))
    assert_size_stride(primals_252, (768, ), (1, ))
    assert_size_stride(primals_253, (768, ), (1, ))
    assert_size_stride(primals_254, (3072, 768), (768, 1))
    assert_size_stride(primals_255, (3072, ), (1, ))
    assert_size_stride(primals_256, (768, 3072), (3072, 1))
    assert_size_stride(primals_257, (768, ), (1, ))
    assert_size_stride(primals_258, (768, ), (1, ))
    assert_size_stride(primals_259, (768, ), (1, ))
    assert_size_stride(primals_260, (384, 768), (768, 1))
    assert_size_stride(primals_261, (384, ), (1, ))
    assert_size_stride(primals_262, (384, 768), (768, 1))
    assert_size_stride(primals_263, (384, ), (1, ))
    assert_size_stride(primals_264, (384, 768), (768, 1))
    assert_size_stride(primals_265, (384, ), (1, ))
    assert_size_stride(primals_266, (768, 1, 9), (9, 9, 1))
    assert_size_stride(primals_267, (384, 768, 1), (768, 1, 1))
    assert_size_stride(primals_268, (54, 384), (384, 1))
    assert_size_stride(primals_269, (54, ), (1, ))
    assert_size_stride(primals_270, (384, 768), (768, 1))
    assert_size_stride(primals_271, (384, ), (1, ))
    assert_size_stride(primals_272, (768, 768), (768, 1))
    assert_size_stride(primals_273, (768, ), (1, ))
    assert_size_stride(primals_274, (768, ), (1, ))
    assert_size_stride(primals_275, (768, ), (1, ))
    assert_size_stride(primals_276, (3072, 768), (768, 1))
    assert_size_stride(primals_277, (3072, ), (1, ))
    assert_size_stride(primals_278, (768, 3072), (3072, 1))
    assert_size_stride(primals_279, (768, ), (1, ))
    assert_size_stride(primals_280, (768, ), (1, ))
    assert_size_stride(primals_281, (768, ), (1, ))
    assert_size_stride(primals_282, (768, 768), (768, 1))
    assert_size_stride(primals_283, (768, ), (1, ))
    assert_size_stride(primals_284, (768, ), (1, ))
    assert_size_stride(primals_285, (768, ), (1, ))
    assert_size_stride(primals_286, (30522, 768), (768, 1))
    assert_size_stride(primals_287, (30522, ), (1, ))
    assert_size_stride(primals_288, (1, 512), (512, 1))
    assert_size_stride(primals_289, (1, 512), (512, 1))
    assert_size_stride(primals_290, (1, 512), (512, 1))
    assert_size_stride(primals_291, (1, 512), (512, 1))
    buf0 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf4 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    buf5 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_embedding_native_layer_norm_0(c_void_p(primals_290.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(primals_289.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(primals_288.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()))
    del primals_13
    del primals_14
    del primals_15
    del primals_17
    # Source Nodes: [embeddings_1, hidden_states], Original ATen: [aten.native_dropout, aten.native_layer_norm]
    buf6 = aten.native_dropout(buf5, 0.1, True)
    buf7 = buf6[0]
    buf8 = buf6[1]
    del buf6
    buf9 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [mixed_query_layer], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_19, reinterpret_tensor(buf7, (512, 768), (768, 1), 0), reinterpret_tensor(primals_18, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf9)
    del primals_19
    buf10 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [mixed_key_layer], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_21, reinterpret_tensor(buf7, (512, 768), (768, 1), 0), reinterpret_tensor(primals_20, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf10)
    del primals_21
    buf11 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [mixed_value_layer], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_23, reinterpret_tensor(buf7, (512, 768), (768, 1), 0), reinterpret_tensor(primals_22, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf11)
    del primals_23
    buf12 = reinterpret_tensor(buf5, (1, 768, 512), (393216, 512, 1), 0); del buf5  # reuse
    cpp_fused_convolution_1(c_void_p(buf7.data_ptr()), c_void_p(buf12.data_ptr()))
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf13 = extern_kernels.convolution(buf12, primals_24, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
    assert_size_stride(buf13, (1, 768, 512), (393216, 512, 1))
    # Source Nodes: [x_1], Original ATen: [aten.convolution]
    buf14 = extern_kernels.convolution(buf13, primals_25, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
    assert_size_stride(buf14, (1, 384, 512), (196608, 512, 1))
    buf15 = empty_strided((512, 384), (1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_mul_view_2(c_void_p(buf14.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf15.data_ptr()))
    buf16 = empty((512, 54), device='cpu', dtype=torch.float32)
    # Source Nodes: [conv_kernel_layer], Original ATen: [aten.mm]
    extern_kernels.mm(buf15, reinterpret_tensor(primals_26, (384, 54), (1, 384), 0), out=buf16)
    buf17 = empty_strided((3072, 1, 1), (1, 3072, 3072), device='cpu', dtype=torch.float32)
    buf18 = reinterpret_tensor(buf16, (3072, 9, 1), (9, 1, 27648), 0); del buf16  # reuse
    buf19 = empty_strided((3072, 1, 1), (1, 3072, 3072), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_3(c_void_p(buf18.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf19.data_ptr()))
    del primals_27
    buf20 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [conv_out_layer], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_29, reinterpret_tensor(buf7, (512, 768), (768, 1), 0), reinterpret_tensor(primals_28, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf20)
    del primals_29
    buf21 = empty((1, 1), device='cpu', dtype=torch.int64)
    buf22 = empty_strided((9, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.int64)
    buf23 = buf18; del buf18  # reuse
    buf24 = empty((1, 512, 384, 9), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_im2col_4(c_void_p(buf23.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf24.data_ptr()))
    buf25 = reinterpret_tensor(buf20, (3072, 64, 1), (64, 1, 1), 0); del buf20  # reuse
    # Source Nodes: [conv_out_layer_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf24, (3072, 64, 9), (576, 9, 1), 0), buf23, out=buf25)
    buf26 = empty((1, 6, 512, 64), device='cpu', dtype=torch.float32)
    buf27 = empty((1, 6, 64, 512), device='cpu', dtype=torch.float32)
    cpp_fused_5(c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf27.data_ptr()))
    buf28 = empty((6, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf26, (6, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf27, (6, 64, 512), (32768, 512, 1), 0), out=buf28)
    buf29 = reinterpret_tensor(buf19, (1, 6, 512, 1), (3072, 512, 1, 3072), 0); del buf19  # reuse
    buf30 = reinterpret_tensor(buf28, (1, 6, 512, 512), (1572864, 262144, 512, 1), 0); del buf28  # reuse
    buf31 = reinterpret_tensor(buf17, (1, 6, 512, 1), (3072, 512, 1, 3072), 0); del buf17  # reuse
    buf32 = buf30; del buf30  # reuse
    cpp_fused_6(c_void_p(buf32.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf31.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf33 = aten.native_dropout(buf32, 0.1, True)
    buf34 = buf33[0]
    buf35 = buf33[1]
    del buf33
    buf36 = empty_strided((1, 6, 512, 512), (1572864, 1, 3072, 6), device='cpu', dtype=torch.bool)
    buf37 = reinterpret_tensor(buf10, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf10  # reuse
    cpp_fused_7(c_void_p(buf35.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()))
    buf38 = reinterpret_tensor(buf11, (6, 512, 64), (32768, 64, 1), 0); del buf11  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf34, (6, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf37, (6, 512, 64), (32768, 64, 1), 0), out=buf38)
    buf39 = empty_strided((1, 6, 512, 512), (1572864, 1, 3072, 6), device='cpu', dtype=torch.float32)
    buf40 = empty((6, 512, 64), device='cpu', dtype=torch.float32)
    buf41 = reinterpret_tensor(buf12, (512, 768), (768, 1), 0); del buf12  # reuse
    cpp_fused_view_8(c_void_p(buf32.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf41.data_ptr()))
    buf42 = reinterpret_tensor(buf0, (512, 768), (768, 1), 0); del buf0  # reuse
    # Source Nodes: [hidden_states_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_31, buf41, reinterpret_tensor(primals_30, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf42)
    del primals_31
    # Source Nodes: [hidden_states_2], Original ATen: [aten.native_dropout]
    buf43 = aten.native_dropout(reinterpret_tensor(buf42, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
    buf44 = buf43[0]
    buf45 = buf43[1]
    del buf43
    buf46 = buf1; del buf1  # reuse
    buf47 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf49 = reinterpret_tensor(buf42, (1, 512, 768), (393216, 768, 1), 0); del buf42  # reuse
    buf50 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_9(c_void_p(buf44.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf50.data_ptr()))
    buf51 = reinterpret_tensor(buf32, (512, 3072), (3072, 1), 0); del buf32  # reuse
    # Source Nodes: [hidden_states_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_35, buf50, reinterpret_tensor(primals_34, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf51)
    del primals_35
    buf52 = empty((512, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_10(c_void_p(buf51.data_ptr()), c_void_p(buf52.data_ptr()))
    buf53 = reinterpret_tensor(buf44, (512, 768), (768, 1), 0); del buf44  # reuse
    # Source Nodes: [hidden_states_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_37, buf52, reinterpret_tensor(primals_36, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf53)
    del primals_37
    # Source Nodes: [hidden_states_7], Original ATen: [aten.native_dropout]
    buf54 = aten.native_dropout(reinterpret_tensor(buf53, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
    buf55 = buf54[0]
    buf56 = buf54[1]
    del buf54
    buf57 = buf46; del buf46  # reuse
    buf58 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf60 = reinterpret_tensor(buf53, (1, 512, 768), (393216, 768, 1), 0); del buf53  # reuse
    buf61 = empty((512, 768), device='cpu', dtype=torch.float32)
    buf65 = empty_strided((1, 768, 512), (393216, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_transpose_view_11(c_void_p(buf55.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf65.data_ptr()))
    del primals_33
    buf62 = reinterpret_tensor(buf38, (512, 384), (384, 1), 0); del buf38  # reuse
    # Source Nodes: [mixed_query_layer_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_41, buf61, reinterpret_tensor(primals_40, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf62)
    del primals_41
    buf63 = reinterpret_tensor(buf27, (512, 384), (384, 1), 0); del buf27  # reuse
    # Source Nodes: [mixed_key_layer_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_43, buf61, reinterpret_tensor(primals_42, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf63)
    del primals_43
    buf64 = reinterpret_tensor(buf25, (512, 384), (384, 1), 0); del buf25  # reuse
    # Source Nodes: [mixed_value_layer_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_45, buf61, reinterpret_tensor(primals_44, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf64)
    del primals_45
    buf66 = reinterpret_tensor(buf55, (1, 768, 512), (393216, 512, 1), 0); del buf55  # reuse
    cpp_fused_convolution_12(c_void_p(buf65.data_ptr()), c_void_p(buf66.data_ptr()))
    # Source Nodes: [x_6], Original ATen: [aten.convolution]
    buf67 = extern_kernels.convolution(buf66, primals_46, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
    assert_size_stride(buf67, (1, 768, 512), (393216, 512, 1))
    # Source Nodes: [x_7], Original ATen: [aten.convolution]
    buf68 = extern_kernels.convolution(buf67, primals_47, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
    assert_size_stride(buf68, (1, 384, 512), (196608, 512, 1))
    buf69 = empty_strided((512, 384), (1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_mul_view_13(c_void_p(buf68.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf69.data_ptr()))
    buf70 = empty((512, 54), device='cpu', dtype=torch.float32)
    # Source Nodes: [conv_kernel_layer_3], Original ATen: [aten.mm]
    extern_kernels.mm(buf69, reinterpret_tensor(primals_48, (384, 54), (1, 384), 0), out=buf70)
    buf71 = reinterpret_tensor(buf31, (3072, 1, 1), (1, 3072, 3072), 0); del buf31  # reuse
    buf72 = reinterpret_tensor(buf70, (3072, 9, 1), (9, 1, 27648), 0); del buf70  # reuse
    buf73 = reinterpret_tensor(buf29, (3072, 1, 1), (1, 3072, 3072), 0); del buf29  # reuse
    cpp_fused__softmax_14(c_void_p(buf72.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf73.data_ptr()))
    del primals_49
    buf74 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [conv_out_layer_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_51, buf61, reinterpret_tensor(primals_50, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf74)
    del primals_51
    buf75 = buf72; del buf72  # reuse
    buf76 = empty((1, 512, 384, 9), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_15(c_void_p(buf75.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf76.data_ptr()))
    buf77 = reinterpret_tensor(buf74, (3072, 64, 1), (64, 1, 1), 0); del buf74  # reuse
    # Source Nodes: [conv_out_layer_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf76, (3072, 64, 9), (576, 9, 1), 0), buf75, out=buf77)
    buf78 = empty((1, 6, 512, 64), device='cpu', dtype=torch.float32)
    buf79 = empty((1, 6, 64, 512), device='cpu', dtype=torch.float32)
    cpp_fused_16(c_void_p(buf62.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf79.data_ptr()))
    buf80 = empty((6, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf78, (6, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf79, (6, 64, 512), (32768, 512, 1), 0), out=buf80)
    buf81 = reinterpret_tensor(buf73, (1, 6, 512, 1), (3072, 512, 1, 3072), 0); del buf73  # reuse
    buf82 = reinterpret_tensor(buf80, (1, 6, 512, 512), (1572864, 262144, 512, 1), 0); del buf80  # reuse
    buf83 = reinterpret_tensor(buf71, (1, 6, 512, 1), (3072, 512, 1, 3072), 0); del buf71  # reuse
    buf84 = buf82; del buf82  # reuse
    cpp_fused_17(c_void_p(buf84.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf83.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf85 = aten.native_dropout(buf84, 0.1, True)
    buf86 = buf85[0]
    buf87 = buf85[1]
    del buf85
    buf88 = reinterpret_tensor(buf35, (1, 6, 512, 512), (1572864, 1, 3072, 6), 0); del buf35  # reuse
    buf89 = reinterpret_tensor(buf63, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf63  # reuse
    cpp_fused_18(c_void_p(buf87.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf89.data_ptr()))
    buf90 = reinterpret_tensor(buf64, (6, 512, 64), (32768, 64, 1), 0); del buf64  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf86, (6, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf89, (6, 512, 64), (32768, 64, 1), 0), out=buf90)
    buf91 = empty_strided((1, 6, 512, 512), (1572864, 1, 3072, 6), device='cpu', dtype=torch.float32)
    buf92 = empty((6, 512, 64), device='cpu', dtype=torch.float32)
    buf93 = reinterpret_tensor(buf66, (512, 768), (768, 1), 0); del buf66  # reuse
    cpp_fused_view_19(c_void_p(buf84.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf93.data_ptr()))
    buf94 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_10], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_53, buf93, reinterpret_tensor(primals_52, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf94)
    del primals_53
    # Source Nodes: [hidden_states_11], Original ATen: [aten.native_dropout]
    buf95 = aten.native_dropout(reinterpret_tensor(buf94, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
    buf96 = buf95[0]
    buf97 = buf95[1]
    del buf95
    buf98 = buf57; del buf57  # reuse
    buf99 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf101 = reinterpret_tensor(buf94, (1, 512, 768), (393216, 768, 1), 0); del buf94  # reuse
    buf102 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_20(c_void_p(buf96.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()))
    del primals_39
    buf103 = reinterpret_tensor(buf84, (512, 3072), (3072, 1), 0); del buf84  # reuse
    # Source Nodes: [hidden_states_13], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_57, buf102, reinterpret_tensor(primals_56, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf103)
    del primals_57
    buf104 = empty((512, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_21(c_void_p(buf103.data_ptr()), c_void_p(buf104.data_ptr()))
    buf105 = reinterpret_tensor(buf96, (512, 768), (768, 1), 0); del buf96  # reuse
    # Source Nodes: [hidden_states_15], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_59, buf104, reinterpret_tensor(primals_58, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf105)
    del primals_59
    # Source Nodes: [hidden_states_16], Original ATen: [aten.native_dropout]
    buf106 = aten.native_dropout(reinterpret_tensor(buf105, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
    buf107 = buf106[0]
    buf108 = buf106[1]
    del buf106
    buf109 = buf98; del buf98  # reuse
    buf110 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf112 = reinterpret_tensor(buf105, (1, 512, 768), (393216, 768, 1), 0); del buf105  # reuse
    buf113 = empty((512, 768), device='cpu', dtype=torch.float32)
    buf117 = empty_strided((1, 768, 512), (393216, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_transpose_view_22(c_void_p(buf107.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(primals_60.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf117.data_ptr()))
    del primals_55
    buf114 = reinterpret_tensor(buf90, (512, 384), (384, 1), 0); del buf90  # reuse
    # Source Nodes: [mixed_query_layer_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_63, buf113, reinterpret_tensor(primals_62, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf114)
    del primals_63
    buf115 = reinterpret_tensor(buf79, (512, 384), (384, 1), 0); del buf79  # reuse
    # Source Nodes: [mixed_key_layer_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_65, buf113, reinterpret_tensor(primals_64, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf115)
    del primals_65
    buf116 = reinterpret_tensor(buf77, (512, 384), (384, 1), 0); del buf77  # reuse
    # Source Nodes: [mixed_value_layer_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_67, buf113, reinterpret_tensor(primals_66, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf116)
    del primals_67
    buf118 = reinterpret_tensor(buf107, (1, 768, 512), (393216, 512, 1), 0); del buf107  # reuse
    cpp_fused_convolution_23(c_void_p(buf117.data_ptr()), c_void_p(buf118.data_ptr()))
    # Source Nodes: [x_12], Original ATen: [aten.convolution]
    buf119 = extern_kernels.convolution(buf118, primals_68, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
    assert_size_stride(buf119, (1, 768, 512), (393216, 512, 1))
    # Source Nodes: [x_13], Original ATen: [aten.convolution]
    buf120 = extern_kernels.convolution(buf119, primals_69, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
    assert_size_stride(buf120, (1, 384, 512), (196608, 512, 1))
    buf121 = empty_strided((512, 384), (1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_mul_view_24(c_void_p(buf120.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf121.data_ptr()))
    buf122 = empty((512, 54), device='cpu', dtype=torch.float32)
    # Source Nodes: [conv_kernel_layer_6], Original ATen: [aten.mm]
    extern_kernels.mm(buf121, reinterpret_tensor(primals_70, (384, 54), (1, 384), 0), out=buf122)
    buf123 = reinterpret_tensor(buf83, (3072, 1, 1), (1, 3072, 3072), 0); del buf83  # reuse
    buf124 = reinterpret_tensor(buf122, (3072, 9, 1), (9, 1, 27648), 0); del buf122  # reuse
    buf125 = reinterpret_tensor(buf81, (3072, 1, 1), (1, 3072, 3072), 0); del buf81  # reuse
    cpp_fused__softmax_25(c_void_p(buf124.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf125.data_ptr()))
    del primals_71
    buf126 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [conv_out_layer_16], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_73, buf113, reinterpret_tensor(primals_72, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf126)
    del primals_73
    buf127 = buf124; del buf124  # reuse
    buf128 = empty((1, 512, 384, 9), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_26(c_void_p(buf127.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf128.data_ptr()))
    buf129 = reinterpret_tensor(buf126, (3072, 64, 1), (64, 1, 1), 0); del buf126  # reuse
    # Source Nodes: [conv_out_layer_22], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf128, (3072, 64, 9), (576, 9, 1), 0), buf127, out=buf129)
    buf130 = empty((1, 6, 512, 64), device='cpu', dtype=torch.float32)
    buf131 = empty((1, 6, 64, 512), device='cpu', dtype=torch.float32)
    cpp_fused_27(c_void_p(buf114.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf131.data_ptr()))
    buf132 = empty((6, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf130, (6, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf131, (6, 64, 512), (32768, 512, 1), 0), out=buf132)
    buf133 = reinterpret_tensor(buf125, (1, 6, 512, 1), (3072, 512, 1, 3072), 0); del buf125  # reuse
    buf134 = reinterpret_tensor(buf132, (1, 6, 512, 512), (1572864, 262144, 512, 1), 0); del buf132  # reuse
    buf135 = reinterpret_tensor(buf123, (1, 6, 512, 1), (3072, 512, 1, 3072), 0); del buf123  # reuse
    buf136 = buf134; del buf134  # reuse
    cpp_fused_28(c_void_p(buf136.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf135.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf137 = aten.native_dropout(buf136, 0.1, True)
    buf138 = buf137[0]
    buf139 = buf137[1]
    del buf137
    buf140 = reinterpret_tensor(buf87, (1, 6, 512, 512), (1572864, 1, 3072, 6), 0); del buf87  # reuse
    buf141 = reinterpret_tensor(buf115, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf115  # reuse
    cpp_fused_29(c_void_p(buf139.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()))
    buf142 = reinterpret_tensor(buf116, (6, 512, 64), (32768, 64, 1), 0); del buf116  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf138, (6, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf141, (6, 512, 64), (32768, 64, 1), 0), out=buf142)
    buf143 = empty_strided((1, 6, 512, 512), (1572864, 1, 3072, 6), device='cpu', dtype=torch.float32)
    buf144 = empty((6, 512, 64), device='cpu', dtype=torch.float32)
    buf145 = reinterpret_tensor(buf118, (512, 768), (768, 1), 0); del buf118  # reuse
    cpp_fused_view_30(c_void_p(buf136.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()))
    buf146 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_19], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_75, buf145, reinterpret_tensor(primals_74, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf146)
    del primals_75
    # Source Nodes: [hidden_states_20], Original ATen: [aten.native_dropout]
    buf147 = aten.native_dropout(reinterpret_tensor(buf146, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
    buf148 = buf147[0]
    buf149 = buf147[1]
    del buf147
    buf150 = buf109; del buf109  # reuse
    buf151 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf153 = reinterpret_tensor(buf146, (1, 512, 768), (393216, 768, 1), 0); del buf146  # reuse
    buf154 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_31(c_void_p(buf148.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(primals_60.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(primals_76.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()))
    del primals_61
    buf155 = reinterpret_tensor(buf136, (512, 3072), (3072, 1), 0); del buf136  # reuse
    # Source Nodes: [hidden_states_22], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_79, buf154, reinterpret_tensor(primals_78, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf155)
    del primals_79
    buf156 = empty((512, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_32(c_void_p(buf155.data_ptr()), c_void_p(buf156.data_ptr()))
    buf157 = reinterpret_tensor(buf148, (512, 768), (768, 1), 0); del buf148  # reuse
    # Source Nodes: [hidden_states_24], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_81, buf156, reinterpret_tensor(primals_80, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf157)
    del primals_81
    # Source Nodes: [hidden_states_25], Original ATen: [aten.native_dropout]
    buf158 = aten.native_dropout(reinterpret_tensor(buf157, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
    buf159 = buf158[0]
    buf160 = buf158[1]
    del buf158
    buf161 = buf150; del buf150  # reuse
    buf162 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf164 = reinterpret_tensor(buf157, (1, 512, 768), (393216, 768, 1), 0); del buf157  # reuse
    buf165 = empty((512, 768), device='cpu', dtype=torch.float32)
    buf169 = empty_strided((1, 768, 512), (393216, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_transpose_view_33(c_void_p(buf159.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(primals_76.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(primals_82.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf169.data_ptr()))
    del primals_77
    buf166 = reinterpret_tensor(buf142, (512, 384), (384, 1), 0); del buf142  # reuse
    # Source Nodes: [mixed_query_layer_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_85, buf165, reinterpret_tensor(primals_84, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf166)
    del primals_85
    buf167 = reinterpret_tensor(buf131, (512, 384), (384, 1), 0); del buf131  # reuse
    # Source Nodes: [mixed_key_layer_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_87, buf165, reinterpret_tensor(primals_86, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf167)
    del primals_87
    buf168 = reinterpret_tensor(buf129, (512, 384), (384, 1), 0); del buf129  # reuse
    # Source Nodes: [mixed_value_layer_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_89, buf165, reinterpret_tensor(primals_88, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf168)
    del primals_89
    buf170 = reinterpret_tensor(buf159, (1, 768, 512), (393216, 512, 1), 0); del buf159  # reuse
    cpp_fused_convolution_34(c_void_p(buf169.data_ptr()), c_void_p(buf170.data_ptr()))
    # Source Nodes: [x_18], Original ATen: [aten.convolution]
    buf171 = extern_kernels.convolution(buf170, primals_90, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
    assert_size_stride(buf171, (1, 768, 512), (393216, 512, 1))
    # Source Nodes: [x_19], Original ATen: [aten.convolution]
    buf172 = extern_kernels.convolution(buf171, primals_91, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
    assert_size_stride(buf172, (1, 384, 512), (196608, 512, 1))
    buf173 = empty_strided((512, 384), (1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_mul_view_35(c_void_p(buf172.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf173.data_ptr()))
    buf174 = empty((512, 54), device='cpu', dtype=torch.float32)
    # Source Nodes: [conv_kernel_layer_9], Original ATen: [aten.mm]
    extern_kernels.mm(buf173, reinterpret_tensor(primals_92, (384, 54), (1, 384), 0), out=buf174)
    buf175 = reinterpret_tensor(buf135, (3072, 1, 1), (1, 3072, 3072), 0); del buf135  # reuse
    buf176 = reinterpret_tensor(buf174, (3072, 9, 1), (9, 1, 27648), 0); del buf174  # reuse
    buf177 = reinterpret_tensor(buf133, (3072, 1, 1), (1, 3072, 3072), 0); del buf133  # reuse
    cpp_fused__softmax_36(c_void_p(buf176.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf177.data_ptr()))
    del primals_93
    buf178 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [conv_out_layer_24], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_95, buf165, reinterpret_tensor(primals_94, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf178)
    del primals_95
    buf179 = buf176; del buf176  # reuse
    buf180 = empty((1, 512, 384, 9), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_37(c_void_p(buf179.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf180.data_ptr()))
    buf181 = reinterpret_tensor(buf178, (3072, 64, 1), (64, 1, 1), 0); del buf178  # reuse
    # Source Nodes: [conv_out_layer_30], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf180, (3072, 64, 9), (576, 9, 1), 0), buf179, out=buf181)
    buf182 = empty((1, 6, 512, 64), device='cpu', dtype=torch.float32)
    buf183 = empty((1, 6, 64, 512), device='cpu', dtype=torch.float32)
    cpp_fused_38(c_void_p(buf166.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf183.data_ptr()))
    buf184 = empty((6, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf182, (6, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf183, (6, 64, 512), (32768, 512, 1), 0), out=buf184)
    buf185 = reinterpret_tensor(buf177, (1, 6, 512, 1), (3072, 512, 1, 3072), 0); del buf177  # reuse
    buf186 = reinterpret_tensor(buf184, (1, 6, 512, 512), (1572864, 262144, 512, 1), 0); del buf184  # reuse
    buf187 = reinterpret_tensor(buf175, (1, 6, 512, 1), (3072, 512, 1, 3072), 0); del buf175  # reuse
    buf188 = buf186; del buf186  # reuse
    cpp_fused_39(c_void_p(buf188.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf187.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf189 = aten.native_dropout(buf188, 0.1, True)
    buf190 = buf189[0]
    buf191 = buf189[1]
    del buf189
    buf192 = reinterpret_tensor(buf139, (1, 6, 512, 512), (1572864, 1, 3072, 6), 0); del buf139  # reuse
    buf193 = reinterpret_tensor(buf167, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf167  # reuse
    cpp_fused_40(c_void_p(buf191.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf193.data_ptr()))
    buf194 = reinterpret_tensor(buf168, (6, 512, 64), (32768, 64, 1), 0); del buf168  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf190, (6, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf193, (6, 512, 64), (32768, 64, 1), 0), out=buf194)
    buf195 = empty_strided((1, 6, 512, 512), (1572864, 1, 3072, 6), device='cpu', dtype=torch.float32)
    buf196 = empty((6, 512, 64), device='cpu', dtype=torch.float32)
    buf197 = reinterpret_tensor(buf170, (512, 768), (768, 1), 0); del buf170  # reuse
    cpp_fused_view_41(c_void_p(buf188.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf197.data_ptr()))
    buf198 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_28], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_97, buf197, reinterpret_tensor(primals_96, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf198)
    del primals_97
    # Source Nodes: [hidden_states_29], Original ATen: [aten.native_dropout]
    buf199 = aten.native_dropout(reinterpret_tensor(buf198, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
    buf200 = buf199[0]
    buf201 = buf199[1]
    del buf199
    buf202 = buf161; del buf161  # reuse
    buf203 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf205 = reinterpret_tensor(buf198, (1, 512, 768), (393216, 768, 1), 0); del buf198  # reuse
    buf206 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_42(c_void_p(buf200.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(primals_82.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(primals_98.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf206.data_ptr()))
    del primals_83
    buf207 = reinterpret_tensor(buf188, (512, 3072), (3072, 1), 0); del buf188  # reuse
    # Source Nodes: [hidden_states_31], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_101, buf206, reinterpret_tensor(primals_100, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf207)
    del primals_101
    buf208 = empty((512, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_43(c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()))
    buf209 = reinterpret_tensor(buf200, (512, 768), (768, 1), 0); del buf200  # reuse
    # Source Nodes: [hidden_states_33], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_103, buf208, reinterpret_tensor(primals_102, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf209)
    del primals_103
    # Source Nodes: [hidden_states_34], Original ATen: [aten.native_dropout]
    buf210 = aten.native_dropout(reinterpret_tensor(buf209, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
    buf211 = buf210[0]
    buf212 = buf210[1]
    del buf210
    buf213 = buf202; del buf202  # reuse
    buf214 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf216 = reinterpret_tensor(buf209, (1, 512, 768), (393216, 768, 1), 0); del buf209  # reuse
    buf217 = empty((512, 768), device='cpu', dtype=torch.float32)
    buf221 = empty_strided((1, 768, 512), (393216, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_transpose_view_44(c_void_p(buf211.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(primals_98.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(primals_105.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf221.data_ptr()))
    del primals_99
    buf218 = reinterpret_tensor(buf194, (512, 384), (384, 1), 0); del buf194  # reuse
    # Source Nodes: [mixed_query_layer_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_107, buf217, reinterpret_tensor(primals_106, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf218)
    del primals_107
    buf219 = reinterpret_tensor(buf183, (512, 384), (384, 1), 0); del buf183  # reuse
    # Source Nodes: [mixed_key_layer_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_109, buf217, reinterpret_tensor(primals_108, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf219)
    del primals_109
    buf220 = reinterpret_tensor(buf181, (512, 384), (384, 1), 0); del buf181  # reuse
    # Source Nodes: [mixed_value_layer_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_111, buf217, reinterpret_tensor(primals_110, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf220)
    del primals_111
    buf222 = reinterpret_tensor(buf211, (1, 768, 512), (393216, 512, 1), 0); del buf211  # reuse
    cpp_fused_convolution_45(c_void_p(buf221.data_ptr()), c_void_p(buf222.data_ptr()))
    # Source Nodes: [x_24], Original ATen: [aten.convolution]
    buf223 = extern_kernels.convolution(buf222, primals_112, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
    assert_size_stride(buf223, (1, 768, 512), (393216, 512, 1))
    # Source Nodes: [x_25], Original ATen: [aten.convolution]
    buf224 = extern_kernels.convolution(buf223, primals_113, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
    assert_size_stride(buf224, (1, 384, 512), (196608, 512, 1))
    buf225 = empty_strided((512, 384), (1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_mul_view_46(c_void_p(buf224.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf225.data_ptr()))
    buf226 = empty((512, 54), device='cpu', dtype=torch.float32)
    # Source Nodes: [conv_kernel_layer_12], Original ATen: [aten.mm]
    extern_kernels.mm(buf225, reinterpret_tensor(primals_114, (384, 54), (1, 384), 0), out=buf226)
    buf227 = reinterpret_tensor(buf187, (3072, 1, 1), (1, 3072, 3072), 0); del buf187  # reuse
    buf228 = reinterpret_tensor(buf226, (3072, 9, 1), (9, 1, 27648), 0); del buf226  # reuse
    buf229 = reinterpret_tensor(buf185, (3072, 1, 1), (1, 3072, 3072), 0); del buf185  # reuse
    cpp_fused__softmax_47(c_void_p(buf228.data_ptr()), c_void_p(primals_115.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf229.data_ptr()))
    del primals_115
    buf230 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [conv_out_layer_32], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_117, buf217, reinterpret_tensor(primals_116, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf230)
    del primals_117
    buf231 = buf228; del buf228  # reuse
    buf232 = empty((1, 512, 384, 9), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_48(c_void_p(buf231.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf232.data_ptr()))
    buf233 = reinterpret_tensor(buf230, (3072, 64, 1), (64, 1, 1), 0); del buf230  # reuse
    # Source Nodes: [conv_out_layer_38], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf232, (3072, 64, 9), (576, 9, 1), 0), buf231, out=buf233)
    buf234 = empty((1, 6, 512, 64), device='cpu', dtype=torch.float32)
    buf235 = empty((1, 6, 64, 512), device='cpu', dtype=torch.float32)
    cpp_fused_49(c_void_p(buf218.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf235.data_ptr()))
    buf236 = empty((6, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf234, (6, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf235, (6, 64, 512), (32768, 512, 1), 0), out=buf236)
    buf237 = reinterpret_tensor(buf229, (1, 6, 512, 1), (3072, 512, 1, 3072), 0); del buf229  # reuse
    buf238 = reinterpret_tensor(buf236, (1, 6, 512, 512), (1572864, 262144, 512, 1), 0); del buf236  # reuse
    buf239 = reinterpret_tensor(buf227, (1, 6, 512, 1), (3072, 512, 1, 3072), 0); del buf227  # reuse
    buf240 = buf238; del buf238  # reuse
    cpp_fused_50(c_void_p(buf240.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf239.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf241 = aten.native_dropout(buf240, 0.1, True)
    buf242 = buf241[0]
    buf243 = buf241[1]
    del buf241
    buf244 = reinterpret_tensor(buf191, (1, 6, 512, 512), (1572864, 1, 3072, 6), 0); del buf191  # reuse
    buf245 = reinterpret_tensor(buf219, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf219  # reuse
    cpp_fused_51(c_void_p(buf243.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf245.data_ptr()))
    buf246 = reinterpret_tensor(buf220, (6, 512, 64), (32768, 64, 1), 0); del buf220  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf242, (6, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf245, (6, 512, 64), (32768, 64, 1), 0), out=buf246)
    buf247 = empty_strided((1, 6, 512, 512), (1572864, 1, 3072, 6), device='cpu', dtype=torch.float32)
    buf248 = empty((6, 512, 64), device='cpu', dtype=torch.float32)
    buf249 = reinterpret_tensor(buf222, (512, 768), (768, 1), 0); del buf222  # reuse
    cpp_fused_view_52(c_void_p(buf240.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf249.data_ptr()))
    buf250 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_37], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_119, buf249, reinterpret_tensor(primals_118, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf250)
    del primals_119
    # Source Nodes: [hidden_states_38], Original ATen: [aten.native_dropout]
    buf251 = aten.native_dropout(reinterpret_tensor(buf250, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
    buf252 = buf251[0]
    buf253 = buf251[1]
    del buf251
    buf254 = buf213; del buf213  # reuse
    buf255 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf257 = reinterpret_tensor(buf250, (1, 512, 768), (393216, 768, 1), 0); del buf250  # reuse
    buf258 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_53(c_void_p(buf252.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(primals_105.data_ptr()), c_void_p(primals_120.data_ptr()), c_void_p(primals_121.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf258.data_ptr()))
    del primals_105
    buf259 = reinterpret_tensor(buf240, (512, 3072), (3072, 1), 0); del buf240  # reuse
    # Source Nodes: [hidden_states_40], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_123, buf258, reinterpret_tensor(primals_122, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf259)
    del primals_123
    buf260 = empty((512, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_54(c_void_p(buf259.data_ptr()), c_void_p(buf260.data_ptr()))
    buf261 = reinterpret_tensor(buf252, (512, 768), (768, 1), 0); del buf252  # reuse
    # Source Nodes: [hidden_states_42], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_125, buf260, reinterpret_tensor(primals_124, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf261)
    del primals_125
    # Source Nodes: [hidden_states_43], Original ATen: [aten.native_dropout]
    buf262 = aten.native_dropout(reinterpret_tensor(buf261, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
    buf263 = buf262[0]
    buf264 = buf262[1]
    del buf262
    buf265 = buf254; del buf254  # reuse
    buf266 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf268 = reinterpret_tensor(buf261, (1, 512, 768), (393216, 768, 1), 0); del buf261  # reuse
    buf269 = empty((512, 768), device='cpu', dtype=torch.float32)
    buf273 = empty_strided((1, 768, 512), (393216, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_transpose_view_55(c_void_p(buf263.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(primals_120.data_ptr()), c_void_p(primals_121.data_ptr()), c_void_p(primals_126.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf273.data_ptr()))
    del primals_121
    buf270 = reinterpret_tensor(buf246, (512, 384), (384, 1), 0); del buf246  # reuse
    # Source Nodes: [mixed_query_layer_5], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_129, buf269, reinterpret_tensor(primals_128, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf270)
    del primals_129
    buf271 = reinterpret_tensor(buf235, (512, 384), (384, 1), 0); del buf235  # reuse
    # Source Nodes: [mixed_key_layer_5], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_131, buf269, reinterpret_tensor(primals_130, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf271)
    del primals_131
    buf272 = reinterpret_tensor(buf233, (512, 384), (384, 1), 0); del buf233  # reuse
    # Source Nodes: [mixed_value_layer_5], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_133, buf269, reinterpret_tensor(primals_132, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf272)
    del primals_133
    buf274 = reinterpret_tensor(buf263, (1, 768, 512), (393216, 512, 1), 0); del buf263  # reuse
    cpp_fused_convolution_56(c_void_p(buf273.data_ptr()), c_void_p(buf274.data_ptr()))
    # Source Nodes: [x_30], Original ATen: [aten.convolution]
    buf275 = extern_kernels.convolution(buf274, primals_134, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
    assert_size_stride(buf275, (1, 768, 512), (393216, 512, 1))
    # Source Nodes: [x_31], Original ATen: [aten.convolution]
    buf276 = extern_kernels.convolution(buf275, primals_135, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
    assert_size_stride(buf276, (1, 384, 512), (196608, 512, 1))
    buf277 = empty_strided((512, 384), (1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_mul_view_57(c_void_p(buf276.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf277.data_ptr()))
    buf278 = empty((512, 54), device='cpu', dtype=torch.float32)
    # Source Nodes: [conv_kernel_layer_15], Original ATen: [aten.mm]
    extern_kernels.mm(buf277, reinterpret_tensor(primals_136, (384, 54), (1, 384), 0), out=buf278)
    buf279 = reinterpret_tensor(buf239, (3072, 1, 1), (1, 3072, 3072), 0); del buf239  # reuse
    buf280 = reinterpret_tensor(buf278, (3072, 9, 1), (9, 1, 27648), 0); del buf278  # reuse
    buf281 = reinterpret_tensor(buf237, (3072, 1, 1), (1, 3072, 3072), 0); del buf237  # reuse
    cpp_fused__softmax_58(c_void_p(buf280.data_ptr()), c_void_p(primals_137.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf281.data_ptr()))
    del primals_137
    buf282 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [conv_out_layer_40], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_139, buf269, reinterpret_tensor(primals_138, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf282)
    del primals_139
    buf283 = buf280; del buf280  # reuse
    buf284 = empty((1, 512, 384, 9), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_59(c_void_p(buf283.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf284.data_ptr()))
    buf285 = reinterpret_tensor(buf282, (3072, 64, 1), (64, 1, 1), 0); del buf282  # reuse
    # Source Nodes: [conv_out_layer_46], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf284, (3072, 64, 9), (576, 9, 1), 0), buf283, out=buf285)
    buf286 = empty((1, 6, 512, 64), device='cpu', dtype=torch.float32)
    buf287 = empty((1, 6, 64, 512), device='cpu', dtype=torch.float32)
    cpp_fused_60(c_void_p(buf270.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf287.data_ptr()))
    buf288 = empty((6, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf286, (6, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf287, (6, 64, 512), (32768, 512, 1), 0), out=buf288)
    buf289 = reinterpret_tensor(buf281, (1, 6, 512, 1), (3072, 512, 1, 3072), 0); del buf281  # reuse
    buf290 = reinterpret_tensor(buf288, (1, 6, 512, 512), (1572864, 262144, 512, 1), 0); del buf288  # reuse
    buf291 = reinterpret_tensor(buf279, (1, 6, 512, 1), (3072, 512, 1, 3072), 0); del buf279  # reuse
    buf292 = buf290; del buf290  # reuse
    cpp_fused_61(c_void_p(buf292.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf291.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf293 = aten.native_dropout(buf292, 0.1, True)
    buf294 = buf293[0]
    buf295 = buf293[1]
    del buf293
    buf296 = reinterpret_tensor(buf243, (1, 6, 512, 512), (1572864, 1, 3072, 6), 0); del buf243  # reuse
    buf297 = reinterpret_tensor(buf271, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf271  # reuse
    cpp_fused_62(c_void_p(buf295.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf297.data_ptr()))
    buf298 = reinterpret_tensor(buf272, (6, 512, 64), (32768, 64, 1), 0); del buf272  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf294, (6, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf297, (6, 512, 64), (32768, 64, 1), 0), out=buf298)
    buf299 = empty_strided((1, 6, 512, 512), (1572864, 1, 3072, 6), device='cpu', dtype=torch.float32)
    buf300 = empty((6, 512, 64), device='cpu', dtype=torch.float32)
    buf301 = reinterpret_tensor(buf274, (512, 768), (768, 1), 0); del buf274  # reuse
    cpp_fused_view_63(c_void_p(buf292.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf301.data_ptr()))
    buf302 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_46], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_141, buf301, reinterpret_tensor(primals_140, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf302)
    del primals_141
    # Source Nodes: [hidden_states_47], Original ATen: [aten.native_dropout]
    buf303 = aten.native_dropout(reinterpret_tensor(buf302, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
    buf304 = buf303[0]
    buf305 = buf303[1]
    del buf303
    buf306 = buf265; del buf265  # reuse
    buf307 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf309 = reinterpret_tensor(buf302, (1, 512, 768), (393216, 768, 1), 0); del buf302  # reuse
    buf310 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_64(c_void_p(buf304.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(primals_126.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(primals_142.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf310.data_ptr()))
    del primals_127
    buf311 = reinterpret_tensor(buf292, (512, 3072), (3072, 1), 0); del buf292  # reuse
    # Source Nodes: [hidden_states_49], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_145, buf310, reinterpret_tensor(primals_144, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf311)
    del primals_145
    buf312 = empty((512, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_65(c_void_p(buf311.data_ptr()), c_void_p(buf312.data_ptr()))
    buf313 = reinterpret_tensor(buf304, (512, 768), (768, 1), 0); del buf304  # reuse
    # Source Nodes: [hidden_states_51], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_147, buf312, reinterpret_tensor(primals_146, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf313)
    del primals_147
    # Source Nodes: [hidden_states_52], Original ATen: [aten.native_dropout]
    buf314 = aten.native_dropout(reinterpret_tensor(buf313, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
    buf315 = buf314[0]
    buf316 = buf314[1]
    del buf314
    buf317 = buf306; del buf306  # reuse
    buf318 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf320 = reinterpret_tensor(buf313, (1, 512, 768), (393216, 768, 1), 0); del buf313  # reuse
    buf321 = empty((512, 768), device='cpu', dtype=torch.float32)
    buf325 = empty_strided((1, 768, 512), (393216, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_transpose_view_66(c_void_p(buf315.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(primals_142.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(primals_148.data_ptr()), c_void_p(primals_149.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf325.data_ptr()))
    del primals_143
    buf322 = reinterpret_tensor(buf298, (512, 384), (384, 1), 0); del buf298  # reuse
    # Source Nodes: [mixed_query_layer_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_151, buf321, reinterpret_tensor(primals_150, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf322)
    del primals_151
    buf323 = reinterpret_tensor(buf287, (512, 384), (384, 1), 0); del buf287  # reuse
    # Source Nodes: [mixed_key_layer_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_153, buf321, reinterpret_tensor(primals_152, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf323)
    del primals_153
    buf324 = reinterpret_tensor(buf285, (512, 384), (384, 1), 0); del buf285  # reuse
    # Source Nodes: [mixed_value_layer_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_155, buf321, reinterpret_tensor(primals_154, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf324)
    del primals_155
    buf326 = reinterpret_tensor(buf315, (1, 768, 512), (393216, 512, 1), 0); del buf315  # reuse
    cpp_fused_convolution_67(c_void_p(buf325.data_ptr()), c_void_p(buf326.data_ptr()))
    # Source Nodes: [x_36], Original ATen: [aten.convolution]
    buf327 = extern_kernels.convolution(buf326, primals_156, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
    assert_size_stride(buf327, (1, 768, 512), (393216, 512, 1))
    # Source Nodes: [x_37], Original ATen: [aten.convolution]
    buf328 = extern_kernels.convolution(buf327, primals_157, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
    assert_size_stride(buf328, (1, 384, 512), (196608, 512, 1))
    buf329 = empty_strided((512, 384), (1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_mul_view_68(c_void_p(buf328.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf329.data_ptr()))
    buf330 = empty((512, 54), device='cpu', dtype=torch.float32)
    # Source Nodes: [conv_kernel_layer_18], Original ATen: [aten.mm]
    extern_kernels.mm(buf329, reinterpret_tensor(primals_158, (384, 54), (1, 384), 0), out=buf330)
    buf331 = reinterpret_tensor(buf291, (3072, 1, 1), (1, 3072, 3072), 0); del buf291  # reuse
    buf332 = reinterpret_tensor(buf330, (3072, 9, 1), (9, 1, 27648), 0); del buf330  # reuse
    buf333 = reinterpret_tensor(buf289, (3072, 1, 1), (1, 3072, 3072), 0); del buf289  # reuse
    cpp_fused__softmax_69(c_void_p(buf332.data_ptr()), c_void_p(primals_159.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf333.data_ptr()))
    del primals_159
    buf334 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [conv_out_layer_48], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_161, buf321, reinterpret_tensor(primals_160, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf334)
    del primals_161
    buf335 = buf332; del buf332  # reuse
    buf336 = empty((1, 512, 384, 9), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_70(c_void_p(buf335.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf336.data_ptr()))
    buf337 = reinterpret_tensor(buf334, (3072, 64, 1), (64, 1, 1), 0); del buf334  # reuse
    # Source Nodes: [conv_out_layer_54], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf336, (3072, 64, 9), (576, 9, 1), 0), buf335, out=buf337)
    buf338 = empty((1, 6, 512, 64), device='cpu', dtype=torch.float32)
    buf339 = empty((1, 6, 64, 512), device='cpu', dtype=torch.float32)
    cpp_fused_71(c_void_p(buf322.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf339.data_ptr()))
    buf340 = empty((6, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf338, (6, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf339, (6, 64, 512), (32768, 512, 1), 0), out=buf340)
    buf341 = reinterpret_tensor(buf333, (1, 6, 512, 1), (3072, 512, 1, 3072), 0); del buf333  # reuse
    buf342 = reinterpret_tensor(buf340, (1, 6, 512, 512), (1572864, 262144, 512, 1), 0); del buf340  # reuse
    buf343 = reinterpret_tensor(buf331, (1, 6, 512, 1), (3072, 512, 1, 3072), 0); del buf331  # reuse
    buf344 = buf342; del buf342  # reuse
    cpp_fused_72(c_void_p(buf344.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf343.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf345 = aten.native_dropout(buf344, 0.1, True)
    buf346 = buf345[0]
    buf347 = buf345[1]
    del buf345
    buf348 = reinterpret_tensor(buf295, (1, 6, 512, 512), (1572864, 1, 3072, 6), 0); del buf295  # reuse
    buf349 = reinterpret_tensor(buf323, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf323  # reuse
    cpp_fused_73(c_void_p(buf347.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf349.data_ptr()))
    buf350 = reinterpret_tensor(buf324, (6, 512, 64), (32768, 64, 1), 0); del buf324  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf346, (6, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf349, (6, 512, 64), (32768, 64, 1), 0), out=buf350)
    buf351 = empty_strided((1, 6, 512, 512), (1572864, 1, 3072, 6), device='cpu', dtype=torch.float32)
    buf352 = empty((6, 512, 64), device='cpu', dtype=torch.float32)
    buf353 = reinterpret_tensor(buf326, (512, 768), (768, 1), 0); del buf326  # reuse
    cpp_fused_view_74(c_void_p(buf344.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf353.data_ptr()))
    buf354 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_55], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_163, buf353, reinterpret_tensor(primals_162, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf354)
    del primals_163
    # Source Nodes: [hidden_states_56], Original ATen: [aten.native_dropout]
    buf355 = aten.native_dropout(reinterpret_tensor(buf354, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
    buf356 = buf355[0]
    buf357 = buf355[1]
    del buf355
    buf358 = buf317; del buf317  # reuse
    buf359 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf361 = reinterpret_tensor(buf354, (1, 512, 768), (393216, 768, 1), 0); del buf354  # reuse
    buf362 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_75(c_void_p(buf356.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(primals_148.data_ptr()), c_void_p(primals_149.data_ptr()), c_void_p(primals_164.data_ptr()), c_void_p(primals_165.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf362.data_ptr()))
    del primals_149
    buf363 = reinterpret_tensor(buf344, (512, 3072), (3072, 1), 0); del buf344  # reuse
    # Source Nodes: [hidden_states_58], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_167, buf362, reinterpret_tensor(primals_166, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf363)
    del primals_167
    buf364 = empty((512, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_76(c_void_p(buf363.data_ptr()), c_void_p(buf364.data_ptr()))
    buf365 = reinterpret_tensor(buf356, (512, 768), (768, 1), 0); del buf356  # reuse
    # Source Nodes: [hidden_states_60], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_169, buf364, reinterpret_tensor(primals_168, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf365)
    del primals_169
    # Source Nodes: [hidden_states_61], Original ATen: [aten.native_dropout]
    buf366 = aten.native_dropout(reinterpret_tensor(buf365, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
    buf367 = buf366[0]
    buf368 = buf366[1]
    del buf366
    buf369 = buf358; del buf358  # reuse
    buf370 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf372 = reinterpret_tensor(buf365, (1, 512, 768), (393216, 768, 1), 0); del buf365  # reuse
    buf373 = empty((512, 768), device='cpu', dtype=torch.float32)
    buf377 = empty_strided((1, 768, 512), (393216, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_transpose_view_77(c_void_p(buf367.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(primals_164.data_ptr()), c_void_p(primals_165.data_ptr()), c_void_p(primals_170.data_ptr()), c_void_p(primals_171.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf377.data_ptr()))
    del primals_165
    buf374 = reinterpret_tensor(buf350, (512, 384), (384, 1), 0); del buf350  # reuse
    # Source Nodes: [mixed_query_layer_7], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_173, buf373, reinterpret_tensor(primals_172, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf374)
    del primals_173
    buf375 = reinterpret_tensor(buf339, (512, 384), (384, 1), 0); del buf339  # reuse
    # Source Nodes: [mixed_key_layer_7], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_175, buf373, reinterpret_tensor(primals_174, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf375)
    del primals_175
    buf376 = reinterpret_tensor(buf337, (512, 384), (384, 1), 0); del buf337  # reuse
    # Source Nodes: [mixed_value_layer_7], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_177, buf373, reinterpret_tensor(primals_176, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf376)
    del primals_177
    buf378 = reinterpret_tensor(buf367, (1, 768, 512), (393216, 512, 1), 0); del buf367  # reuse
    cpp_fused_convolution_78(c_void_p(buf377.data_ptr()), c_void_p(buf378.data_ptr()))
    # Source Nodes: [x_42], Original ATen: [aten.convolution]
    buf379 = extern_kernels.convolution(buf378, primals_178, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
    assert_size_stride(buf379, (1, 768, 512), (393216, 512, 1))
    # Source Nodes: [x_43], Original ATen: [aten.convolution]
    buf380 = extern_kernels.convolution(buf379, primals_179, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
    assert_size_stride(buf380, (1, 384, 512), (196608, 512, 1))
    buf381 = empty_strided((512, 384), (1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_mul_view_79(c_void_p(buf380.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf381.data_ptr()))
    buf382 = empty((512, 54), device='cpu', dtype=torch.float32)
    # Source Nodes: [conv_kernel_layer_21], Original ATen: [aten.mm]
    extern_kernels.mm(buf381, reinterpret_tensor(primals_180, (384, 54), (1, 384), 0), out=buf382)
    buf383 = reinterpret_tensor(buf343, (3072, 1, 1), (1, 3072, 3072), 0); del buf343  # reuse
    buf384 = reinterpret_tensor(buf382, (3072, 9, 1), (9, 1, 27648), 0); del buf382  # reuse
    buf385 = reinterpret_tensor(buf341, (3072, 1, 1), (1, 3072, 3072), 0); del buf341  # reuse
    cpp_fused__softmax_80(c_void_p(buf384.data_ptr()), c_void_p(primals_181.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(buf385.data_ptr()))
    del primals_181
    buf386 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [conv_out_layer_56], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_183, buf373, reinterpret_tensor(primals_182, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf386)
    del primals_183
    buf387 = buf384; del buf384  # reuse
    buf388 = empty((1, 512, 384, 9), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_81(c_void_p(buf387.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf388.data_ptr()))
    buf389 = reinterpret_tensor(buf386, (3072, 64, 1), (64, 1, 1), 0); del buf386  # reuse
    # Source Nodes: [conv_out_layer_62], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf388, (3072, 64, 9), (576, 9, 1), 0), buf387, out=buf389)
    buf390 = empty((1, 6, 512, 64), device='cpu', dtype=torch.float32)
    buf391 = empty((1, 6, 64, 512), device='cpu', dtype=torch.float32)
    cpp_fused_82(c_void_p(buf374.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(buf391.data_ptr()))
    buf392 = empty((6, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf390, (6, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf391, (6, 64, 512), (32768, 512, 1), 0), out=buf392)
    buf393 = reinterpret_tensor(buf385, (1, 6, 512, 1), (3072, 512, 1, 3072), 0); del buf385  # reuse
    buf394 = reinterpret_tensor(buf392, (1, 6, 512, 512), (1572864, 262144, 512, 1), 0); del buf392  # reuse
    buf395 = reinterpret_tensor(buf383, (1, 6, 512, 1), (3072, 512, 1, 3072), 0); del buf383  # reuse
    buf396 = buf394; del buf394  # reuse
    cpp_fused_83(c_void_p(buf396.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf395.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf397 = aten.native_dropout(buf396, 0.1, True)
    buf398 = buf397[0]
    buf399 = buf397[1]
    del buf397
    buf400 = reinterpret_tensor(buf347, (1, 6, 512, 512), (1572864, 1, 3072, 6), 0); del buf347  # reuse
    buf401 = reinterpret_tensor(buf375, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf375  # reuse
    cpp_fused_84(c_void_p(buf399.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(buf401.data_ptr()))
    buf402 = reinterpret_tensor(buf376, (6, 512, 64), (32768, 64, 1), 0); del buf376  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf398, (6, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf401, (6, 512, 64), (32768, 64, 1), 0), out=buf402)
    buf403 = empty_strided((1, 6, 512, 512), (1572864, 1, 3072, 6), device='cpu', dtype=torch.float32)
    buf404 = empty((6, 512, 64), device='cpu', dtype=torch.float32)
    buf405 = reinterpret_tensor(buf378, (512, 768), (768, 1), 0); del buf378  # reuse
    cpp_fused_view_85(c_void_p(buf396.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(buf405.data_ptr()))
    buf406 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_64], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_185, buf405, reinterpret_tensor(primals_184, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf406)
    del primals_185
    # Source Nodes: [hidden_states_65], Original ATen: [aten.native_dropout]
    buf407 = aten.native_dropout(reinterpret_tensor(buf406, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
    buf408 = buf407[0]
    buf409 = buf407[1]
    del buf407
    buf410 = buf369; del buf369  # reuse
    buf411 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf413 = reinterpret_tensor(buf406, (1, 512, 768), (393216, 768, 1), 0); del buf406  # reuse
    buf414 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_86(c_void_p(buf408.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(primals_170.data_ptr()), c_void_p(primals_171.data_ptr()), c_void_p(primals_186.data_ptr()), c_void_p(primals_187.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(buf411.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(buf414.data_ptr()))
    del primals_171
    buf415 = reinterpret_tensor(buf396, (512, 3072), (3072, 1), 0); del buf396  # reuse
    # Source Nodes: [hidden_states_67], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_189, buf414, reinterpret_tensor(primals_188, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf415)
    del primals_189
    buf416 = empty((512, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_87(c_void_p(buf415.data_ptr()), c_void_p(buf416.data_ptr()))
    buf417 = reinterpret_tensor(buf408, (512, 768), (768, 1), 0); del buf408  # reuse
    # Source Nodes: [hidden_states_69], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_191, buf416, reinterpret_tensor(primals_190, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf417)
    del primals_191
    # Source Nodes: [hidden_states_70], Original ATen: [aten.native_dropout]
    buf418 = aten.native_dropout(reinterpret_tensor(buf417, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
    buf419 = buf418[0]
    buf420 = buf418[1]
    del buf418
    buf421 = buf410; del buf410  # reuse
    buf422 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf424 = reinterpret_tensor(buf417, (1, 512, 768), (393216, 768, 1), 0); del buf417  # reuse
    buf425 = empty((512, 768), device='cpu', dtype=torch.float32)
    buf429 = empty_strided((1, 768, 512), (393216, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_transpose_view_88(c_void_p(buf419.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(primals_186.data_ptr()), c_void_p(primals_187.data_ptr()), c_void_p(primals_192.data_ptr()), c_void_p(primals_193.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf429.data_ptr()))
    del primals_187
    buf426 = reinterpret_tensor(buf402, (512, 384), (384, 1), 0); del buf402  # reuse
    # Source Nodes: [mixed_query_layer_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_195, buf425, reinterpret_tensor(primals_194, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf426)
    del primals_195
    buf427 = reinterpret_tensor(buf391, (512, 384), (384, 1), 0); del buf391  # reuse
    # Source Nodes: [mixed_key_layer_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_197, buf425, reinterpret_tensor(primals_196, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf427)
    del primals_197
    buf428 = reinterpret_tensor(buf389, (512, 384), (384, 1), 0); del buf389  # reuse
    # Source Nodes: [mixed_value_layer_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_199, buf425, reinterpret_tensor(primals_198, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf428)
    del primals_199
    buf430 = reinterpret_tensor(buf419, (1, 768, 512), (393216, 512, 1), 0); del buf419  # reuse
    cpp_fused_convolution_89(c_void_p(buf429.data_ptr()), c_void_p(buf430.data_ptr()))
    # Source Nodes: [x_48], Original ATen: [aten.convolution]
    buf431 = extern_kernels.convolution(buf430, primals_200, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
    assert_size_stride(buf431, (1, 768, 512), (393216, 512, 1))
    # Source Nodes: [x_49], Original ATen: [aten.convolution]
    buf432 = extern_kernels.convolution(buf431, primals_201, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
    assert_size_stride(buf432, (1, 384, 512), (196608, 512, 1))
    buf433 = empty_strided((512, 384), (1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_mul_view_90(c_void_p(buf432.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf433.data_ptr()))
    buf434 = empty((512, 54), device='cpu', dtype=torch.float32)
    # Source Nodes: [conv_kernel_layer_24], Original ATen: [aten.mm]
    extern_kernels.mm(buf433, reinterpret_tensor(primals_202, (384, 54), (1, 384), 0), out=buf434)
    buf435 = reinterpret_tensor(buf395, (3072, 1, 1), (1, 3072, 3072), 0); del buf395  # reuse
    buf436 = reinterpret_tensor(buf434, (3072, 9, 1), (9, 1, 27648), 0); del buf434  # reuse
    buf437 = reinterpret_tensor(buf393, (3072, 1, 1), (1, 3072, 3072), 0); del buf393  # reuse
    cpp_fused__softmax_91(c_void_p(buf436.data_ptr()), c_void_p(primals_203.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf437.data_ptr()))
    del primals_203
    buf438 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [conv_out_layer_64], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_205, buf425, reinterpret_tensor(primals_204, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf438)
    del primals_205
    buf439 = buf436; del buf436  # reuse
    buf440 = empty((1, 512, 384, 9), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_92(c_void_p(buf439.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(buf440.data_ptr()))
    buf441 = reinterpret_tensor(buf438, (3072, 64, 1), (64, 1, 1), 0); del buf438  # reuse
    # Source Nodes: [conv_out_layer_70], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf440, (3072, 64, 9), (576, 9, 1), 0), buf439, out=buf441)
    buf442 = empty((1, 6, 512, 64), device='cpu', dtype=torch.float32)
    buf443 = empty((1, 6, 64, 512), device='cpu', dtype=torch.float32)
    cpp_fused_93(c_void_p(buf426.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(buf443.data_ptr()))
    buf444 = empty((6, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf442, (6, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf443, (6, 64, 512), (32768, 512, 1), 0), out=buf444)
    buf445 = reinterpret_tensor(buf437, (1, 6, 512, 1), (3072, 512, 1, 3072), 0); del buf437  # reuse
    buf446 = reinterpret_tensor(buf444, (1, 6, 512, 512), (1572864, 262144, 512, 1), 0); del buf444  # reuse
    buf447 = reinterpret_tensor(buf435, (1, 6, 512, 1), (3072, 512, 1, 3072), 0); del buf435  # reuse
    buf448 = buf446; del buf446  # reuse
    cpp_fused_94(c_void_p(buf448.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(buf447.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf449 = aten.native_dropout(buf448, 0.1, True)
    buf450 = buf449[0]
    buf451 = buf449[1]
    del buf449
    buf452 = reinterpret_tensor(buf399, (1, 6, 512, 512), (1572864, 1, 3072, 6), 0); del buf399  # reuse
    buf453 = reinterpret_tensor(buf427, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf427  # reuse
    cpp_fused_95(c_void_p(buf451.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf453.data_ptr()))
    buf454 = reinterpret_tensor(buf428, (6, 512, 64), (32768, 64, 1), 0); del buf428  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf450, (6, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf453, (6, 512, 64), (32768, 64, 1), 0), out=buf454)
    buf455 = empty_strided((1, 6, 512, 512), (1572864, 1, 3072, 6), device='cpu', dtype=torch.float32)
    buf456 = empty((6, 512, 64), device='cpu', dtype=torch.float32)
    buf457 = reinterpret_tensor(buf430, (512, 768), (768, 1), 0); del buf430  # reuse
    cpp_fused_view_96(c_void_p(buf448.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf454.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(buf457.data_ptr()))
    buf458 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_73], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_207, buf457, reinterpret_tensor(primals_206, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf458)
    del primals_207
    # Source Nodes: [hidden_states_74], Original ATen: [aten.native_dropout]
    buf459 = aten.native_dropout(reinterpret_tensor(buf458, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
    buf460 = buf459[0]
    buf461 = buf459[1]
    del buf459
    buf462 = buf421; del buf421  # reuse
    buf463 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf465 = reinterpret_tensor(buf458, (1, 512, 768), (393216, 768, 1), 0); del buf458  # reuse
    buf466 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_97(c_void_p(buf460.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(primals_192.data_ptr()), c_void_p(primals_193.data_ptr()), c_void_p(primals_208.data_ptr()), c_void_p(primals_209.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(buf463.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(buf466.data_ptr()))
    del primals_193
    buf467 = reinterpret_tensor(buf448, (512, 3072), (3072, 1), 0); del buf448  # reuse
    # Source Nodes: [hidden_states_76], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_211, buf466, reinterpret_tensor(primals_210, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf467)
    del primals_211
    buf468 = empty((512, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_98(c_void_p(buf467.data_ptr()), c_void_p(buf468.data_ptr()))
    buf469 = reinterpret_tensor(buf460, (512, 768), (768, 1), 0); del buf460  # reuse
    # Source Nodes: [hidden_states_78], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_213, buf468, reinterpret_tensor(primals_212, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf469)
    del primals_213
    # Source Nodes: [hidden_states_79], Original ATen: [aten.native_dropout]
    buf470 = aten.native_dropout(reinterpret_tensor(buf469, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
    buf471 = buf470[0]
    buf472 = buf470[1]
    del buf470
    buf473 = buf462; del buf462  # reuse
    buf474 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf476 = reinterpret_tensor(buf469, (1, 512, 768), (393216, 768, 1), 0); del buf469  # reuse
    buf477 = empty((512, 768), device='cpu', dtype=torch.float32)
    buf481 = empty_strided((1, 768, 512), (393216, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_transpose_view_99(c_void_p(buf471.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(primals_208.data_ptr()), c_void_p(primals_209.data_ptr()), c_void_p(primals_214.data_ptr()), c_void_p(primals_215.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(buf474.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(buf477.data_ptr()), c_void_p(buf481.data_ptr()))
    del primals_209
    buf478 = reinterpret_tensor(buf454, (512, 384), (384, 1), 0); del buf454  # reuse
    # Source Nodes: [mixed_query_layer_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_217, buf477, reinterpret_tensor(primals_216, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf478)
    del primals_217
    buf479 = reinterpret_tensor(buf443, (512, 384), (384, 1), 0); del buf443  # reuse
    # Source Nodes: [mixed_key_layer_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_219, buf477, reinterpret_tensor(primals_218, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf479)
    del primals_219
    buf480 = reinterpret_tensor(buf441, (512, 384), (384, 1), 0); del buf441  # reuse
    # Source Nodes: [mixed_value_layer_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_221, buf477, reinterpret_tensor(primals_220, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf480)
    del primals_221
    buf482 = reinterpret_tensor(buf471, (1, 768, 512), (393216, 512, 1), 0); del buf471  # reuse
    cpp_fused_convolution_100(c_void_p(buf481.data_ptr()), c_void_p(buf482.data_ptr()))
    # Source Nodes: [x_54], Original ATen: [aten.convolution]
    buf483 = extern_kernels.convolution(buf482, primals_222, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
    assert_size_stride(buf483, (1, 768, 512), (393216, 512, 1))
    # Source Nodes: [x_55], Original ATen: [aten.convolution]
    buf484 = extern_kernels.convolution(buf483, primals_223, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
    assert_size_stride(buf484, (1, 384, 512), (196608, 512, 1))
    buf485 = empty_strided((512, 384), (1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_mul_view_101(c_void_p(buf484.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(buf478.data_ptr()), c_void_p(buf485.data_ptr()))
    buf486 = empty((512, 54), device='cpu', dtype=torch.float32)
    # Source Nodes: [conv_kernel_layer_27], Original ATen: [aten.mm]
    extern_kernels.mm(buf485, reinterpret_tensor(primals_224, (384, 54), (1, 384), 0), out=buf486)
    buf487 = reinterpret_tensor(buf447, (3072, 1, 1), (1, 3072, 3072), 0); del buf447  # reuse
    buf488 = reinterpret_tensor(buf486, (3072, 9, 1), (9, 1, 27648), 0); del buf486  # reuse
    buf489 = reinterpret_tensor(buf445, (3072, 1, 1), (1, 3072, 3072), 0); del buf445  # reuse
    cpp_fused__softmax_102(c_void_p(buf488.data_ptr()), c_void_p(primals_225.data_ptr()), c_void_p(buf487.data_ptr()), c_void_p(buf489.data_ptr()))
    del primals_225
    buf490 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [conv_out_layer_72], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_227, buf477, reinterpret_tensor(primals_226, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf490)
    del primals_227
    buf491 = buf488; del buf488  # reuse
    buf492 = empty((1, 512, 384, 9), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_103(c_void_p(buf491.data_ptr()), c_void_p(buf489.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf490.data_ptr()), c_void_p(buf492.data_ptr()))
    buf493 = reinterpret_tensor(buf490, (3072, 64, 1), (64, 1, 1), 0); del buf490  # reuse
    # Source Nodes: [conv_out_layer_78], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf492, (3072, 64, 9), (576, 9, 1), 0), buf491, out=buf493)
    buf494 = empty((1, 6, 512, 64), device='cpu', dtype=torch.float32)
    buf495 = empty((1, 6, 64, 512), device='cpu', dtype=torch.float32)
    cpp_fused_104(c_void_p(buf478.data_ptr()), c_void_p(buf479.data_ptr()), c_void_p(buf494.data_ptr()), c_void_p(buf495.data_ptr()))
    buf496 = empty((6, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf494, (6, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf495, (6, 64, 512), (32768, 512, 1), 0), out=buf496)
    buf497 = reinterpret_tensor(buf489, (1, 6, 512, 1), (3072, 512, 1, 3072), 0); del buf489  # reuse
    buf498 = reinterpret_tensor(buf496, (1, 6, 512, 512), (1572864, 262144, 512, 1), 0); del buf496  # reuse
    buf499 = reinterpret_tensor(buf487, (1, 6, 512, 1), (3072, 512, 1, 3072), 0); del buf487  # reuse
    buf500 = buf498; del buf498  # reuse
    cpp_fused_105(c_void_p(buf500.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(buf499.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf501 = aten.native_dropout(buf500, 0.1, True)
    buf502 = buf501[0]
    buf503 = buf501[1]
    del buf501
    buf504 = reinterpret_tensor(buf451, (1, 6, 512, 512), (1572864, 1, 3072, 6), 0); del buf451  # reuse
    buf505 = reinterpret_tensor(buf479, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf479  # reuse
    cpp_fused_106(c_void_p(buf503.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(buf505.data_ptr()))
    buf506 = reinterpret_tensor(buf480, (6, 512, 64), (32768, 64, 1), 0); del buf480  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf502, (6, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf505, (6, 512, 64), (32768, 64, 1), 0), out=buf506)
    buf507 = empty_strided((1, 6, 512, 512), (1572864, 1, 3072, 6), device='cpu', dtype=torch.float32)
    buf508 = empty((6, 512, 64), device='cpu', dtype=torch.float32)
    buf509 = reinterpret_tensor(buf482, (512, 768), (768, 1), 0); del buf482  # reuse
    cpp_fused_view_107(c_void_p(buf500.data_ptr()), c_void_p(buf495.data_ptr()), c_void_p(buf506.data_ptr()), c_void_p(buf493.data_ptr()), c_void_p(buf507.data_ptr()), c_void_p(buf508.data_ptr()), c_void_p(buf509.data_ptr()))
    buf510 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_82], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_229, buf509, reinterpret_tensor(primals_228, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf510)
    del primals_229
    # Source Nodes: [hidden_states_83], Original ATen: [aten.native_dropout]
    buf511 = aten.native_dropout(reinterpret_tensor(buf510, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
    buf512 = buf511[0]
    buf513 = buf511[1]
    del buf511
    buf514 = buf473; del buf473  # reuse
    buf515 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf517 = reinterpret_tensor(buf510, (1, 512, 768), (393216, 768, 1), 0); del buf510  # reuse
    buf518 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_108(c_void_p(buf512.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(primals_214.data_ptr()), c_void_p(primals_215.data_ptr()), c_void_p(primals_230.data_ptr()), c_void_p(primals_231.data_ptr()), c_void_p(buf514.data_ptr()), c_void_p(buf515.data_ptr()), c_void_p(buf517.data_ptr()), c_void_p(buf518.data_ptr()))
    del primals_215
    buf519 = reinterpret_tensor(buf500, (512, 3072), (3072, 1), 0); del buf500  # reuse
    # Source Nodes: [hidden_states_85], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_233, buf518, reinterpret_tensor(primals_232, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf519)
    del primals_233
    buf520 = empty((512, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_109(c_void_p(buf519.data_ptr()), c_void_p(buf520.data_ptr()))
    buf521 = reinterpret_tensor(buf512, (512, 768), (768, 1), 0); del buf512  # reuse
    # Source Nodes: [hidden_states_87], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_235, buf520, reinterpret_tensor(primals_234, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf521)
    del primals_235
    # Source Nodes: [hidden_states_88], Original ATen: [aten.native_dropout]
    buf522 = aten.native_dropout(reinterpret_tensor(buf521, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
    buf523 = buf522[0]
    buf524 = buf522[1]
    del buf522
    buf525 = buf514; del buf514  # reuse
    buf526 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf528 = reinterpret_tensor(buf521, (1, 512, 768), (393216, 768, 1), 0); del buf521  # reuse
    buf529 = empty((512, 768), device='cpu', dtype=torch.float32)
    buf533 = empty_strided((1, 768, 512), (393216, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_transpose_view_110(c_void_p(buf523.data_ptr()), c_void_p(buf517.data_ptr()), c_void_p(primals_230.data_ptr()), c_void_p(primals_231.data_ptr()), c_void_p(primals_236.data_ptr()), c_void_p(primals_237.data_ptr()), c_void_p(buf525.data_ptr()), c_void_p(buf526.data_ptr()), c_void_p(buf528.data_ptr()), c_void_p(buf529.data_ptr()), c_void_p(buf533.data_ptr()))
    del primals_231
    buf530 = reinterpret_tensor(buf506, (512, 384), (384, 1), 0); del buf506  # reuse
    # Source Nodes: [mixed_query_layer_10], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_239, buf529, reinterpret_tensor(primals_238, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf530)
    del primals_239
    buf531 = reinterpret_tensor(buf495, (512, 384), (384, 1), 0); del buf495  # reuse
    # Source Nodes: [mixed_key_layer_10], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_241, buf529, reinterpret_tensor(primals_240, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf531)
    del primals_241
    buf532 = reinterpret_tensor(buf493, (512, 384), (384, 1), 0); del buf493  # reuse
    # Source Nodes: [mixed_value_layer_10], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_243, buf529, reinterpret_tensor(primals_242, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf532)
    del primals_243
    buf534 = reinterpret_tensor(buf523, (1, 768, 512), (393216, 512, 1), 0); del buf523  # reuse
    cpp_fused_convolution_111(c_void_p(buf533.data_ptr()), c_void_p(buf534.data_ptr()))
    # Source Nodes: [x_60], Original ATen: [aten.convolution]
    buf535 = extern_kernels.convolution(buf534, primals_244, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
    assert_size_stride(buf535, (1, 768, 512), (393216, 512, 1))
    # Source Nodes: [x_61], Original ATen: [aten.convolution]
    buf536 = extern_kernels.convolution(buf535, primals_245, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
    assert_size_stride(buf536, (1, 384, 512), (196608, 512, 1))
    buf537 = empty_strided((512, 384), (1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_mul_view_112(c_void_p(buf536.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf530.data_ptr()), c_void_p(buf537.data_ptr()))
    buf538 = empty((512, 54), device='cpu', dtype=torch.float32)
    # Source Nodes: [conv_kernel_layer_30], Original ATen: [aten.mm]
    extern_kernels.mm(buf537, reinterpret_tensor(primals_246, (384, 54), (1, 384), 0), out=buf538)
    buf539 = reinterpret_tensor(buf499, (3072, 1, 1), (1, 3072, 3072), 0); del buf499  # reuse
    buf540 = reinterpret_tensor(buf538, (3072, 9, 1), (9, 1, 27648), 0); del buf538  # reuse
    buf541 = reinterpret_tensor(buf497, (3072, 1, 1), (1, 3072, 3072), 0); del buf497  # reuse
    cpp_fused__softmax_113(c_void_p(buf540.data_ptr()), c_void_p(primals_247.data_ptr()), c_void_p(buf539.data_ptr()), c_void_p(buf541.data_ptr()))
    del primals_247
    buf542 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [conv_out_layer_80], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_249, buf529, reinterpret_tensor(primals_248, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf542)
    del primals_249
    buf543 = buf540; del buf540  # reuse
    buf544 = empty((1, 512, 384, 9), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_114(c_void_p(buf543.data_ptr()), c_void_p(buf541.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf542.data_ptr()), c_void_p(buf544.data_ptr()))
    buf545 = reinterpret_tensor(buf542, (3072, 64, 1), (64, 1, 1), 0); del buf542  # reuse
    # Source Nodes: [conv_out_layer_86], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf544, (3072, 64, 9), (576, 9, 1), 0), buf543, out=buf545)
    buf546 = empty((1, 6, 512, 64), device='cpu', dtype=torch.float32)
    buf547 = empty((1, 6, 64, 512), device='cpu', dtype=torch.float32)
    cpp_fused_115(c_void_p(buf530.data_ptr()), c_void_p(buf531.data_ptr()), c_void_p(buf546.data_ptr()), c_void_p(buf547.data_ptr()))
    buf548 = empty((6, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf546, (6, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf547, (6, 64, 512), (32768, 512, 1), 0), out=buf548)
    buf549 = reinterpret_tensor(buf541, (1, 6, 512, 1), (3072, 512, 1, 3072), 0); del buf541  # reuse
    buf550 = reinterpret_tensor(buf548, (1, 6, 512, 512), (1572864, 262144, 512, 1), 0); del buf548  # reuse
    buf551 = reinterpret_tensor(buf539, (1, 6, 512, 1), (3072, 512, 1, 3072), 0); del buf539  # reuse
    buf552 = buf550; del buf550  # reuse
    cpp_fused_116(c_void_p(buf552.data_ptr()), c_void_p(buf549.data_ptr()), c_void_p(buf551.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf553 = aten.native_dropout(buf552, 0.1, True)
    buf554 = buf553[0]
    buf555 = buf553[1]
    del buf553
    buf556 = reinterpret_tensor(buf503, (1, 6, 512, 512), (1572864, 1, 3072, 6), 0); del buf503  # reuse
    buf557 = reinterpret_tensor(buf531, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf531  # reuse
    cpp_fused_117(c_void_p(buf555.data_ptr()), c_void_p(buf532.data_ptr()), c_void_p(buf556.data_ptr()), c_void_p(buf557.data_ptr()))
    buf558 = reinterpret_tensor(buf532, (6, 512, 64), (32768, 64, 1), 0); del buf532  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf554, (6, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf557, (6, 512, 64), (32768, 64, 1), 0), out=buf558)
    buf559 = empty_strided((1, 6, 512, 512), (1572864, 1, 3072, 6), device='cpu', dtype=torch.float32)
    buf560 = empty((6, 512, 64), device='cpu', dtype=torch.float32)
    buf561 = reinterpret_tensor(buf534, (512, 768), (768, 1), 0); del buf534  # reuse
    cpp_fused_view_118(c_void_p(buf552.data_ptr()), c_void_p(buf547.data_ptr()), c_void_p(buf558.data_ptr()), c_void_p(buf545.data_ptr()), c_void_p(buf559.data_ptr()), c_void_p(buf560.data_ptr()), c_void_p(buf561.data_ptr()))
    buf562 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_91], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_251, buf561, reinterpret_tensor(primals_250, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf562)
    del primals_251
    # Source Nodes: [hidden_states_92], Original ATen: [aten.native_dropout]
    buf563 = aten.native_dropout(reinterpret_tensor(buf562, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
    buf564 = buf563[0]
    buf565 = buf563[1]
    del buf563
    buf566 = buf525; del buf525  # reuse
    buf567 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf569 = reinterpret_tensor(buf562, (1, 512, 768), (393216, 768, 1), 0); del buf562  # reuse
    buf570 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_119(c_void_p(buf564.data_ptr()), c_void_p(buf528.data_ptr()), c_void_p(primals_236.data_ptr()), c_void_p(primals_237.data_ptr()), c_void_p(primals_252.data_ptr()), c_void_p(primals_253.data_ptr()), c_void_p(buf566.data_ptr()), c_void_p(buf567.data_ptr()), c_void_p(buf569.data_ptr()), c_void_p(buf570.data_ptr()))
    del primals_237
    buf571 = reinterpret_tensor(buf552, (512, 3072), (3072, 1), 0); del buf552  # reuse
    # Source Nodes: [hidden_states_94], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_255, buf570, reinterpret_tensor(primals_254, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf571)
    del primals_255
    buf572 = empty((512, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_120(c_void_p(buf571.data_ptr()), c_void_p(buf572.data_ptr()))
    buf573 = reinterpret_tensor(buf564, (512, 768), (768, 1), 0); del buf564  # reuse
    # Source Nodes: [hidden_states_96], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_257, buf572, reinterpret_tensor(primals_256, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf573)
    del primals_257
    # Source Nodes: [hidden_states_97], Original ATen: [aten.native_dropout]
    buf574 = aten.native_dropout(reinterpret_tensor(buf573, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
    buf575 = buf574[0]
    buf576 = buf574[1]
    del buf574
    buf577 = buf566; del buf566  # reuse
    buf578 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf580 = reinterpret_tensor(buf573, (1, 512, 768), (393216, 768, 1), 0); del buf573  # reuse
    buf581 = empty((512, 768), device='cpu', dtype=torch.float32)
    buf585 = empty_strided((1, 768, 512), (393216, 1, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_transpose_view_121(c_void_p(buf575.data_ptr()), c_void_p(buf569.data_ptr()), c_void_p(primals_252.data_ptr()), c_void_p(primals_253.data_ptr()), c_void_p(primals_258.data_ptr()), c_void_p(primals_259.data_ptr()), c_void_p(buf577.data_ptr()), c_void_p(buf578.data_ptr()), c_void_p(buf580.data_ptr()), c_void_p(buf581.data_ptr()), c_void_p(buf585.data_ptr()))
    del primals_253
    buf582 = reinterpret_tensor(buf558, (512, 384), (384, 1), 0); del buf558  # reuse
    # Source Nodes: [mixed_query_layer_11], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_261, buf581, reinterpret_tensor(primals_260, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf582)
    del primals_261
    buf583 = reinterpret_tensor(buf547, (512, 384), (384, 1), 0); del buf547  # reuse
    # Source Nodes: [mixed_key_layer_11], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_263, buf581, reinterpret_tensor(primals_262, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf583)
    del primals_263
    buf584 = reinterpret_tensor(buf545, (512, 384), (384, 1), 0); del buf545  # reuse
    # Source Nodes: [mixed_value_layer_11], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_265, buf581, reinterpret_tensor(primals_264, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf584)
    del primals_265
    buf586 = reinterpret_tensor(buf575, (1, 768, 512), (393216, 512, 1), 0); del buf575  # reuse
    cpp_fused_convolution_122(c_void_p(buf585.data_ptr()), c_void_p(buf586.data_ptr()))
    # Source Nodes: [x_66], Original ATen: [aten.convolution]
    buf587 = extern_kernels.convolution(buf586, primals_266, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=768, bias=None)
    assert_size_stride(buf587, (1, 768, 512), (393216, 512, 1))
    # Source Nodes: [x_67], Original ATen: [aten.convolution]
    buf588 = extern_kernels.convolution(buf587, primals_267, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
    assert_size_stride(buf588, (1, 384, 512), (196608, 512, 1))
    buf589 = empty_strided((512, 384), (1, 512), device='cpu', dtype=torch.float32)
    cpp_fused_mul_view_123(c_void_p(buf588.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(buf582.data_ptr()), c_void_p(buf589.data_ptr()))
    buf590 = empty((512, 54), device='cpu', dtype=torch.float32)
    # Source Nodes: [conv_kernel_layer_33], Original ATen: [aten.mm]
    extern_kernels.mm(buf589, reinterpret_tensor(primals_268, (384, 54), (1, 384), 0), out=buf590)
    buf591 = reinterpret_tensor(buf551, (3072, 1, 1), (1, 3072, 3072), 0); del buf551  # reuse
    buf592 = reinterpret_tensor(buf590, (3072, 9, 1), (9, 1, 27648), 0); del buf590  # reuse
    buf593 = reinterpret_tensor(buf549, (3072, 1, 1), (1, 3072, 3072), 0); del buf549  # reuse
    cpp_fused__softmax_124(c_void_p(buf592.data_ptr()), c_void_p(primals_269.data_ptr()), c_void_p(buf591.data_ptr()), c_void_p(buf593.data_ptr()))
    del primals_269
    buf594 = empty((512, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [conv_out_layer_88], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_271, buf581, reinterpret_tensor(primals_270, (768, 384), (1, 768), 0), alpha=1, beta=1, out=buf594)
    del primals_271
    buf595 = buf592; del buf592  # reuse
    buf596 = empty((1, 512, 384, 9), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_125(c_void_p(buf595.data_ptr()), c_void_p(buf593.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf594.data_ptr()), c_void_p(buf596.data_ptr()))
    buf597 = reinterpret_tensor(buf594, (3072, 64, 1), (64, 1, 1), 0); del buf594  # reuse
    # Source Nodes: [conv_out_layer_94], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf596, (3072, 64, 9), (576, 9, 1), 0), buf595, out=buf597)
    buf598 = empty((1, 6, 512, 64), device='cpu', dtype=torch.float32)
    buf599 = empty((1, 6, 64, 512), device='cpu', dtype=torch.float32)
    cpp_fused_126(c_void_p(buf582.data_ptr()), c_void_p(buf583.data_ptr()), c_void_p(buf598.data_ptr()), c_void_p(buf599.data_ptr()))
    buf600 = empty((6, 512, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf598, (6, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf599, (6, 64, 512), (32768, 512, 1), 0), out=buf600)
    buf601 = reinterpret_tensor(buf593, (1, 6, 512, 1), (3072, 512, 1, 3072), 0); del buf593  # reuse
    buf602 = reinterpret_tensor(buf600, (1, 6, 512, 512), (1572864, 262144, 512, 1), 0); del buf600  # reuse
    buf603 = reinterpret_tensor(buf591, (1, 6, 512, 1), (3072, 512, 1, 3072), 0); del buf591  # reuse
    buf604 = buf602; del buf602  # reuse
    cpp_fused_127(c_void_p(buf604.data_ptr()), c_void_p(buf601.data_ptr()), c_void_p(buf603.data_ptr()))
    del buf601
    del buf603
    # Source Nodes: [], Original ATen: []
    buf605 = aten.native_dropout(buf604, 0.1, True)
    buf606 = buf605[0]
    buf607 = buf605[1]
    del buf605
    buf608 = reinterpret_tensor(buf555, (1, 6, 512, 512), (1572864, 1, 3072, 6), 0); del buf555  # reuse
    buf609 = reinterpret_tensor(buf583, (1, 6, 512, 64), (196608, 32768, 64, 1), 0); del buf583  # reuse
    cpp_fused_128(c_void_p(buf607.data_ptr()), c_void_p(buf584.data_ptr()), c_void_p(buf608.data_ptr()), c_void_p(buf609.data_ptr()))
    del buf607
    buf610 = reinterpret_tensor(buf584, (6, 512, 64), (32768, 64, 1), 0); del buf584  # reuse
    # Source Nodes: [], Original ATen: []
    extern_kernels.bmm(reinterpret_tensor(buf606, (6, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf609, (6, 512, 64), (32768, 64, 1), 0), out=buf610)
    buf611 = empty_strided((1, 6, 512, 512), (1572864, 1, 3072, 6), device='cpu', dtype=torch.float32)
    buf612 = empty((6, 512, 64), device='cpu', dtype=torch.float32)
    buf613 = reinterpret_tensor(buf586, (512, 768), (768, 1), 0); del buf586  # reuse
    cpp_fused_view_129(c_void_p(buf604.data_ptr()), c_void_p(buf599.data_ptr()), c_void_p(buf610.data_ptr()), c_void_p(buf597.data_ptr()), c_void_p(buf611.data_ptr()), c_void_p(buf612.data_ptr()), c_void_p(buf613.data_ptr()))
    del buf597
    del buf599
    del buf610
    buf614 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_100], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_273, buf613, reinterpret_tensor(primals_272, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf614)
    del primals_273
    # Source Nodes: [hidden_states_101], Original ATen: [aten.native_dropout]
    buf615 = aten.native_dropout(reinterpret_tensor(buf614, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
    buf616 = buf615[0]
    buf617 = buf615[1]
    del buf615
    buf618 = buf577; del buf577  # reuse
    buf619 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf621 = reinterpret_tensor(buf614, (1, 512, 768), (393216, 768, 1), 0); del buf614  # reuse
    buf622 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_130(c_void_p(buf616.data_ptr()), c_void_p(buf580.data_ptr()), c_void_p(primals_258.data_ptr()), c_void_p(primals_259.data_ptr()), c_void_p(primals_274.data_ptr()), c_void_p(primals_275.data_ptr()), c_void_p(buf618.data_ptr()), c_void_p(buf619.data_ptr()), c_void_p(buf621.data_ptr()), c_void_p(buf622.data_ptr()))
    del primals_259
    buf623 = reinterpret_tensor(buf604, (512, 3072), (3072, 1), 0); del buf604  # reuse
    # Source Nodes: [hidden_states_103], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_277, buf622, reinterpret_tensor(primals_276, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf623)
    del primals_277
    buf624 = empty((512, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_131(c_void_p(buf623.data_ptr()), c_void_p(buf624.data_ptr()))
    buf625 = reinterpret_tensor(buf616, (512, 768), (768, 1), 0); del buf616  # reuse
    # Source Nodes: [hidden_states_105], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_279, buf624, reinterpret_tensor(primals_278, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf625)
    del primals_279
    # Source Nodes: [hidden_states_106], Original ATen: [aten.native_dropout]
    buf626 = aten.native_dropout(reinterpret_tensor(buf625, (1, 512, 768), (393216, 768, 1), 0), 0.1, True)
    buf627 = buf626[0]
    buf628 = buf626[1]
    del buf626
    buf629 = buf618; del buf618  # reuse
    buf630 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf632 = reinterpret_tensor(buf625, (1, 512, 768), (393216, 768, 1), 0); del buf625  # reuse
    buf633 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_132(c_void_p(buf627.data_ptr()), c_void_p(buf621.data_ptr()), c_void_p(primals_274.data_ptr()), c_void_p(primals_275.data_ptr()), c_void_p(primals_280.data_ptr()), c_void_p(primals_281.data_ptr()), c_void_p(buf629.data_ptr()), c_void_p(buf630.data_ptr()), c_void_p(buf632.data_ptr()), c_void_p(buf633.data_ptr()))
    del primals_275
    del primals_281
    buf634 = reinterpret_tensor(buf627, (512, 768), (768, 1), 0); del buf627  # reuse
    # Source Nodes: [hidden_states_109], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_283, buf633, reinterpret_tensor(primals_282, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf634)
    del primals_283
    buf635 = buf629; del buf629  # reuse
    buf636 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf638 = empty((1, 512, 768), device='cpu', dtype=torch.float32)
    buf639 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_native_layer_norm_view_133(c_void_p(buf634.data_ptr()), c_void_p(primals_284.data_ptr()), c_void_p(primals_285.data_ptr()), c_void_p(buf635.data_ptr()), c_void_p(buf636.data_ptr()), c_void_p(buf638.data_ptr()), c_void_p(buf639.data_ptr()))
    del primals_285
    buf640 = empty((512, 30522), device='cpu', dtype=torch.float32)
    # Source Nodes: [prediction_scores_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_287, buf639, reinterpret_tensor(primals_286, (768, 30522), (1, 768), 0), alpha=1, beta=1, out=buf640)
    del primals_287
    buf641 = reinterpret_tensor(buf635, (512, 1), (1, 512), 0); del buf635  # reuse
    buf642 = empty_strided((512, 1), (1, 512), device='cpu', dtype=torch.float32)
    buf643 = empty((512, 30522), device='cpu', dtype=torch.float32)
    buf644 = empty((), device='cpu', dtype=torch.int64)
    buf646 = empty((), device='cpu', dtype=torch.float32)
    buf645 = empty((), device='cpu', dtype=torch.float32)
    buf673 = buf646; del buf646  # reuse
    buf647 = reinterpret_tensor(buf636, (1, 512, 1), (512, 1, 1), 0); del buf636  # reuse
    buf648 = reinterpret_tensor(buf630, (1, 512, 1), (512, 1, 1), 0); del buf630  # reuse
    buf649 = reinterpret_tensor(buf619, (1, 512, 1), (512, 1, 1), 0); del buf619  # reuse
    buf650 = reinterpret_tensor(buf578, (1, 512, 1), (512, 1, 1), 0); del buf578  # reuse
    buf651 = reinterpret_tensor(buf567, (1, 512, 1), (512, 1, 1), 0); del buf567  # reuse
    buf652 = reinterpret_tensor(buf526, (1, 512, 1), (512, 1, 1), 0); del buf526  # reuse
    buf653 = reinterpret_tensor(buf515, (1, 512, 1), (512, 1, 1), 0); del buf515  # reuse
    buf654 = reinterpret_tensor(buf474, (1, 512, 1), (512, 1, 1), 0); del buf474  # reuse
    buf655 = reinterpret_tensor(buf463, (1, 512, 1), (512, 1, 1), 0); del buf463  # reuse
    buf656 = reinterpret_tensor(buf422, (1, 512, 1), (512, 1, 1), 0); del buf422  # reuse
    buf657 = reinterpret_tensor(buf411, (1, 512, 1), (512, 1, 1), 0); del buf411  # reuse
    buf658 = reinterpret_tensor(buf370, (1, 512, 1), (512, 1, 1), 0); del buf370  # reuse
    buf659 = reinterpret_tensor(buf359, (1, 512, 1), (512, 1, 1), 0); del buf359  # reuse
    buf660 = reinterpret_tensor(buf318, (1, 512, 1), (512, 1, 1), 0); del buf318  # reuse
    buf661 = reinterpret_tensor(buf307, (1, 512, 1), (512, 1, 1), 0); del buf307  # reuse
    buf662 = reinterpret_tensor(buf266, (1, 512, 1), (512, 1, 1), 0); del buf266  # reuse
    buf663 = reinterpret_tensor(buf255, (1, 512, 1), (512, 1, 1), 0); del buf255  # reuse
    buf664 = reinterpret_tensor(buf214, (1, 512, 1), (512, 1, 1), 0); del buf214  # reuse
    buf665 = reinterpret_tensor(buf203, (1, 512, 1), (512, 1, 1), 0); del buf203  # reuse
    buf666 = reinterpret_tensor(buf162, (1, 512, 1), (512, 1, 1), 0); del buf162  # reuse
    buf667 = reinterpret_tensor(buf151, (1, 512, 1), (512, 1, 1), 0); del buf151  # reuse
    buf668 = reinterpret_tensor(buf110, (1, 512, 1), (512, 1, 1), 0); del buf110  # reuse
    buf669 = reinterpret_tensor(buf99, (1, 512, 1), (512, 1, 1), 0); del buf99  # reuse
    buf670 = reinterpret_tensor(buf58, (1, 512, 1), (512, 1, 1), 0); del buf58  # reuse
    buf671 = reinterpret_tensor(buf47, (1, 512, 1), (512, 1, 1), 0); del buf47  # reuse
    buf672 = reinterpret_tensor(buf2, (1, 512, 1), (512, 1, 1), 0); del buf2  # reuse
    cpp_fused__log_softmax_add_gelu_native_layer_norm_native_layer_norm_backward_nll_loss_forward_134(c_void_p(buf673.data_ptr()), c_void_p(buf647.data_ptr()), c_void_p(buf648.data_ptr()), c_void_p(buf649.data_ptr()), c_void_p(buf650.data_ptr()), c_void_p(buf651.data_ptr()), c_void_p(buf652.data_ptr()), c_void_p(buf653.data_ptr()), c_void_p(buf654.data_ptr()), c_void_p(buf655.data_ptr()), c_void_p(buf656.data_ptr()), c_void_p(buf657.data_ptr()), c_void_p(buf658.data_ptr()), c_void_p(buf659.data_ptr()), c_void_p(buf660.data_ptr()), c_void_p(buf661.data_ptr()), c_void_p(buf662.data_ptr()), c_void_p(buf663.data_ptr()), c_void_p(buf664.data_ptr()), c_void_p(buf665.data_ptr()), c_void_p(buf666.data_ptr()), c_void_p(buf667.data_ptr()), c_void_p(buf668.data_ptr()), c_void_p(buf669.data_ptr()), c_void_p(buf670.data_ptr()), c_void_p(buf671.data_ptr()), c_void_p(buf672.data_ptr()), c_void_p(buf640.data_ptr()), c_void_p(primals_291.data_ptr()), c_void_p(buf641.data_ptr()), c_void_p(buf642.data_ptr()), c_void_p(buf643.data_ptr()), c_void_p(buf644.data_ptr()), c_void_p(buf645.data_ptr()))
    return (buf673, reinterpret_tensor(buf640, (1, 512, 30522), (15627264, 30522, 1), 0), primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_16, primals_24, primals_25, primals_32, primals_38, primals_46, primals_47, primals_54, primals_60, primals_68, primals_69, primals_76, primals_82, primals_90, primals_91, primals_98, primals_104, primals_112, primals_113, primals_120, primals_126, primals_134, primals_135, primals_142, primals_148, primals_156, primals_157, primals_164, primals_170, primals_178, primals_179, primals_186, primals_192, primals_200, primals_201, primals_208, primals_214, primals_222, primals_223, primals_230, primals_236, primals_244, primals_245, primals_252, primals_258, primals_266, primals_267, primals_274, primals_280, primals_284, primals_290, primals_291, primals_288, primals_289, buf4, buf8, reinterpret_tensor(buf7, (512, 768), (768, 1), 0), buf9, reinterpret_tensor(buf7, (1, 768, 512), (393216, 1, 768), 0), buf13, buf14, reinterpret_tensor(primals_26, (384, 54), (1, 384), 0), buf15, buf21, buf22, buf36, reinterpret_tensor(buf34, (6, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf37, (6, 64, 512), (32768, 1, 64), 0), buf39, reinterpret_tensor(buf26, (6, 64, 512), (32768, 1, 64), 0), buf40, buf41, buf45, buf49, buf50, buf51, buf52, buf56, buf60, buf61, buf62, buf65, buf67, buf68, reinterpret_tensor(primals_48, (384, 54), (1, 384), 0), buf69, buf88, reinterpret_tensor(buf86, (6, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf89, (6, 64, 512), (32768, 1, 64), 0), buf91, reinterpret_tensor(buf78, (6, 64, 512), (32768, 1, 64), 0), buf92, buf93, buf97, buf101, buf102, buf103, buf104, buf108, buf112, buf113, buf114, buf117, buf119, buf120, reinterpret_tensor(primals_70, (384, 54), (1, 384), 0), buf121, buf140, reinterpret_tensor(buf138, (6, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf141, (6, 64, 512), (32768, 1, 64), 0), buf143, reinterpret_tensor(buf130, (6, 64, 512), (32768, 1, 64), 0), buf144, buf145, buf149, buf153, buf154, buf155, buf156, buf160, buf164, buf165, buf166, buf169, buf171, buf172, reinterpret_tensor(primals_92, (384, 54), (1, 384), 0), buf173, buf192, reinterpret_tensor(buf190, (6, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf193, (6, 64, 512), (32768, 1, 64), 0), buf195, reinterpret_tensor(buf182, (6, 64, 512), (32768, 1, 64), 0), buf196, buf197, buf201, buf205, buf206, buf207, buf208, buf212, buf216, buf217, buf218, buf221, buf223, buf224, reinterpret_tensor(primals_114, (384, 54), (1, 384), 0), buf225, buf244, reinterpret_tensor(buf242, (6, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf245, (6, 64, 512), (32768, 1, 64), 0), buf247, reinterpret_tensor(buf234, (6, 64, 512), (32768, 1, 64), 0), buf248, buf249, buf253, buf257, buf258, buf259, buf260, buf264, buf268, buf269, buf270, buf273, buf275, buf276, reinterpret_tensor(primals_136, (384, 54), (1, 384), 0), buf277, buf296, reinterpret_tensor(buf294, (6, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf297, (6, 64, 512), (32768, 1, 64), 0), buf299, reinterpret_tensor(buf286, (6, 64, 512), (32768, 1, 64), 0), buf300, buf301, buf305, buf309, buf310, buf311, buf312, buf316, buf320, buf321, buf322, buf325, buf327, buf328, reinterpret_tensor(primals_158, (384, 54), (1, 384), 0), buf329, buf348, reinterpret_tensor(buf346, (6, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf349, (6, 64, 512), (32768, 1, 64), 0), buf351, reinterpret_tensor(buf338, (6, 64, 512), (32768, 1, 64), 0), buf352, buf353, buf357, buf361, buf362, buf363, buf364, buf368, buf372, buf373, buf374, buf377, buf379, buf380, reinterpret_tensor(primals_180, (384, 54), (1, 384), 0), buf381, buf400, reinterpret_tensor(buf398, (6, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf401, (6, 64, 512), (32768, 1, 64), 0), buf403, reinterpret_tensor(buf390, (6, 64, 512), (32768, 1, 64), 0), buf404, buf405, buf409, buf413, buf414, buf415, buf416, buf420, buf424, buf425, buf426, buf429, buf431, buf432, reinterpret_tensor(primals_202, (384, 54), (1, 384), 0), buf433, buf452, reinterpret_tensor(buf450, (6, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf453, (6, 64, 512), (32768, 1, 64), 0), buf455, reinterpret_tensor(buf442, (6, 64, 512), (32768, 1, 64), 0), buf456, buf457, buf461, buf465, buf466, buf467, buf468, buf472, buf476, buf477, buf478, buf481, buf483, buf484, reinterpret_tensor(primals_224, (384, 54), (1, 384), 0), buf485, buf504, reinterpret_tensor(buf502, (6, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf505, (6, 64, 512), (32768, 1, 64), 0), buf507, reinterpret_tensor(buf494, (6, 64, 512), (32768, 1, 64), 0), buf508, buf509, buf513, buf517, buf518, buf519, buf520, buf524, buf528, buf529, buf530, buf533, buf535, buf536, reinterpret_tensor(primals_246, (384, 54), (1, 384), 0), buf537, buf556, reinterpret_tensor(buf554, (6, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf557, (6, 64, 512), (32768, 1, 64), 0), buf559, reinterpret_tensor(buf546, (6, 64, 512), (32768, 1, 64), 0), buf560, buf561, buf565, buf569, buf570, buf571, buf572, buf576, buf580, buf581, buf582, buf585, buf587, buf588, reinterpret_tensor(primals_268, (384, 54), (1, 384), 0), buf589, buf608, reinterpret_tensor(buf606, (6, 512, 512), (262144, 1, 512), 0), reinterpret_tensor(buf609, (6, 64, 512), (32768, 1, 64), 0), buf611, reinterpret_tensor(buf598, (6, 64, 512), (32768, 1, 64), 0), buf612, buf613, buf617, buf621, buf622, buf623, buf624, buf628, buf632, buf633, buf634, buf638, buf639, buf643, buf645, reinterpret_tensor(primals_286, (30522, 768), (768, 1), 0), buf647, reinterpret_tensor(primals_282, (768, 768), (768, 1), 0), buf648, reinterpret_tensor(primals_278, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_276, (3072, 768), (768, 1), 0), buf649, reinterpret_tensor(primals_272, (768, 768), (768, 1), 0), reinterpret_tensor(buf596, (3072, 9, 64), (576, 1, 9), 0), reinterpret_tensor(buf595, (3072, 1, 9), (9, 27648, 1), 0), reinterpret_tensor(primals_270, (384, 768), (768, 1), 0), buf595, reinterpret_tensor(primals_264, (384, 768), (768, 1), 0), reinterpret_tensor(primals_262, (384, 768), (768, 1), 0), reinterpret_tensor(primals_260, (384, 768), (768, 1), 0), buf650, reinterpret_tensor(primals_256, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_254, (3072, 768), (768, 1), 0), buf651, reinterpret_tensor(primals_250, (768, 768), (768, 1), 0), reinterpret_tensor(buf544, (3072, 9, 64), (576, 1, 9), 0), reinterpret_tensor(buf543, (3072, 1, 9), (9, 27648, 1), 0), reinterpret_tensor(primals_248, (384, 768), (768, 1), 0), buf543, reinterpret_tensor(primals_242, (384, 768), (768, 1), 0), reinterpret_tensor(primals_240, (384, 768), (768, 1), 0), reinterpret_tensor(primals_238, (384, 768), (768, 1), 0), buf652, reinterpret_tensor(primals_234, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_232, (3072, 768), (768, 1), 0), buf653, reinterpret_tensor(primals_228, (768, 768), (768, 1), 0), reinterpret_tensor(buf492, (3072, 9, 64), (576, 1, 9), 0), reinterpret_tensor(buf491, (3072, 1, 9), (9, 27648, 1), 0), reinterpret_tensor(primals_226, (384, 768), (768, 1), 0), buf491, reinterpret_tensor(primals_220, (384, 768), (768, 1), 0), reinterpret_tensor(primals_218, (384, 768), (768, 1), 0), reinterpret_tensor(primals_216, (384, 768), (768, 1), 0), buf654, reinterpret_tensor(primals_212, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_210, (3072, 768), (768, 1), 0), buf655, reinterpret_tensor(primals_206, (768, 768), (768, 1), 0), reinterpret_tensor(buf440, (3072, 9, 64), (576, 1, 9), 0), reinterpret_tensor(buf439, (3072, 1, 9), (9, 27648, 1), 0), reinterpret_tensor(primals_204, (384, 768), (768, 1), 0), buf439, reinterpret_tensor(primals_198, (384, 768), (768, 1), 0), reinterpret_tensor(primals_196, (384, 768), (768, 1), 0), reinterpret_tensor(primals_194, (384, 768), (768, 1), 0), buf656, reinterpret_tensor(primals_190, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_188, (3072, 768), (768, 1), 0), buf657, reinterpret_tensor(primals_184, (768, 768), (768, 1), 0), reinterpret_tensor(buf388, (3072, 9, 64), (576, 1, 9), 0), reinterpret_tensor(buf387, (3072, 1, 9), (9, 27648, 1), 0), reinterpret_tensor(primals_182, (384, 768), (768, 1), 0), buf387, reinterpret_tensor(primals_176, (384, 768), (768, 1), 0), reinterpret_tensor(primals_174, (384, 768), (768, 1), 0), reinterpret_tensor(primals_172, (384, 768), (768, 1), 0), buf658, reinterpret_tensor(primals_168, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_166, (3072, 768), (768, 1), 0), buf659, reinterpret_tensor(primals_162, (768, 768), (768, 1), 0), reinterpret_tensor(buf336, (3072, 9, 64), (576, 1, 9), 0), reinterpret_tensor(buf335, (3072, 1, 9), (9, 27648, 1), 0), reinterpret_tensor(primals_160, (384, 768), (768, 1), 0), buf335, reinterpret_tensor(primals_154, (384, 768), (768, 1), 0), reinterpret_tensor(primals_152, (384, 768), (768, 1), 0), reinterpret_tensor(primals_150, (384, 768), (768, 1), 0), buf660, reinterpret_tensor(primals_146, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_144, (3072, 768), (768, 1), 0), buf661, reinterpret_tensor(primals_140, (768, 768), (768, 1), 0), reinterpret_tensor(buf284, (3072, 9, 64), (576, 1, 9), 0), reinterpret_tensor(buf283, (3072, 1, 9), (9, 27648, 1), 0), reinterpret_tensor(primals_138, (384, 768), (768, 1), 0), buf283, reinterpret_tensor(primals_132, (384, 768), (768, 1), 0), reinterpret_tensor(primals_130, (384, 768), (768, 1), 0), reinterpret_tensor(primals_128, (384, 768), (768, 1), 0), buf662, reinterpret_tensor(primals_124, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_122, (3072, 768), (768, 1), 0), buf663, reinterpret_tensor(primals_118, (768, 768), (768, 1), 0), reinterpret_tensor(buf232, (3072, 9, 64), (576, 1, 9), 0), reinterpret_tensor(buf231, (3072, 1, 9), (9, 27648, 1), 0), reinterpret_tensor(primals_116, (384, 768), (768, 1), 0), buf231, reinterpret_tensor(primals_110, (384, 768), (768, 1), 0), reinterpret_tensor(primals_108, (384, 768), (768, 1), 0), reinterpret_tensor(primals_106, (384, 768), (768, 1), 0), buf664, reinterpret_tensor(primals_102, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_100, (3072, 768), (768, 1), 0), buf665, reinterpret_tensor(primals_96, (768, 768), (768, 1), 0), reinterpret_tensor(buf180, (3072, 9, 64), (576, 1, 9), 0), reinterpret_tensor(buf179, (3072, 1, 9), (9, 27648, 1), 0), reinterpret_tensor(primals_94, (384, 768), (768, 1), 0), buf179, reinterpret_tensor(primals_88, (384, 768), (768, 1), 0), reinterpret_tensor(primals_86, (384, 768), (768, 1), 0), reinterpret_tensor(primals_84, (384, 768), (768, 1), 0), buf666, reinterpret_tensor(primals_80, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_78, (3072, 768), (768, 1), 0), buf667, reinterpret_tensor(primals_74, (768, 768), (768, 1), 0), reinterpret_tensor(buf128, (3072, 9, 64), (576, 1, 9), 0), reinterpret_tensor(buf127, (3072, 1, 9), (9, 27648, 1), 0), reinterpret_tensor(primals_72, (384, 768), (768, 1), 0), buf127, reinterpret_tensor(primals_66, (384, 768), (768, 1), 0), reinterpret_tensor(primals_64, (384, 768), (768, 1), 0), reinterpret_tensor(primals_62, (384, 768), (768, 1), 0), buf668, reinterpret_tensor(primals_58, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_56, (3072, 768), (768, 1), 0), buf669, reinterpret_tensor(primals_52, (768, 768), (768, 1), 0), reinterpret_tensor(buf76, (3072, 9, 64), (576, 1, 9), 0), reinterpret_tensor(buf75, (3072, 1, 9), (9, 27648, 1), 0), reinterpret_tensor(primals_50, (384, 768), (768, 1), 0), buf75, reinterpret_tensor(primals_44, (384, 768), (768, 1), 0), reinterpret_tensor(primals_42, (384, 768), (768, 1), 0), reinterpret_tensor(primals_40, (384, 768), (768, 1), 0), buf670, reinterpret_tensor(primals_36, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_34, (3072, 768), (768, 1), 0), buf671, reinterpret_tensor(primals_30, (768, 768), (768, 1), 0), reinterpret_tensor(buf24, (3072, 9, 64), (576, 1, 9), 0), reinterpret_tensor(buf23, (3072, 1, 9), (9, 27648, 1), 0), reinterpret_tensor(primals_28, (384, 768), (768, 1), 0), buf23, reinterpret_tensor(primals_22, (384, 768), (768, 1), 0), reinterpret_tensor(primals_20, (384, 768), (768, 1), 0), reinterpret_tensor(primals_18, (384, 768), (768, 1), 0), buf672, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((384, 1), (1, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((384, 1), (1, 1), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((384, 1), (1, 1), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((384, 1), (1, 1), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((384, 1), (1, 1), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((384, 1), (1, 1), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((384, 1), (1, 1), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((384, 1), (1, 1), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((384, 1), (1, 1), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((384, 1), (1, 1), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((384, 1), (1, 1), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((384, 1), (1, 1), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((30522, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((2, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((768, 1, 9), (9, 9, 1), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((384, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((54, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((54, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((768, 1, 9), (9, 9, 1), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((384, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((54, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((54, ), (1, ), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((768, 1, 9), (9, 9, 1), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((384, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((54, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((54, ), (1, ), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((768, 1, 9), (9, 9, 1), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((384, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((54, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((54, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((768, 1, 9), (9, 9, 1), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((384, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((54, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((54, ), (1, ), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((768, 1, 9), (9, 9, 1), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((384, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((54, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((54, ), (1, ), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((768, 1, 9), (9, 9, 1), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((384, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((54, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((54, ), (1, ), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((768, 1, 9), (9, 9, 1), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((384, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((54, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((54, ), (1, ), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((768, 1, 9), (9, 9, 1), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((384, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((54, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((54, ), (1, ), device='cpu', dtype=torch.float32)
    primals_204 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_206 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_207 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_209 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_210 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_212 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_213 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_214 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_215 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_216 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_218 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_219 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_220 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_221 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_222 = rand_strided((768, 1, 9), (9, 9, 1), device='cpu', dtype=torch.float32)
    primals_223 = rand_strided((384, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    primals_224 = rand_strided((54, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_225 = rand_strided((54, ), (1, ), device='cpu', dtype=torch.float32)
    primals_226 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_227 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_228 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_229 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_230 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_231 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_232 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_233 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_234 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_235 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_236 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_237 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_238 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_239 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_240 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_241 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_242 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_243 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_244 = rand_strided((768, 1, 9), (9, 9, 1), device='cpu', dtype=torch.float32)
    primals_245 = rand_strided((384, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    primals_246 = rand_strided((54, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_247 = rand_strided((54, ), (1, ), device='cpu', dtype=torch.float32)
    primals_248 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_249 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_250 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_251 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_252 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_253 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_254 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_255 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_256 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_257 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_258 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_259 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_260 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_261 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_262 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_263 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_264 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_265 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_266 = rand_strided((768, 1, 9), (9, 9, 1), device='cpu', dtype=torch.float32)
    primals_267 = rand_strided((384, 768, 1), (768, 1, 1), device='cpu', dtype=torch.float32)
    primals_268 = rand_strided((54, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_269 = rand_strided((54, ), (1, ), device='cpu', dtype=torch.float32)
    primals_270 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_271 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_272 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_273 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_274 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_275 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_276 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_277 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_278 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_279 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_280 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_281 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_282 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_283 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_284 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_285 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_286 = rand_strided((30522, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_287 = rand_strided((30522, ), (1, ), device='cpu', dtype=torch.float32)
    primals_288 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    primals_289 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    primals_290 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    primals_291 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('YituTechConvBert', benchmark_compiled_module)
