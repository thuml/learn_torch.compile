
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


cpp_fused_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x2 + (16L*x1) + (48L*x0))];
                    out_ptr0[static_cast<long>(x1 + (3L*x2) + (48L*x0))] = tmp0;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (256L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (64L*x2) + (256L*x0)));
                }
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (512L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr2 + static_cast<long>(x1 + (128L*x2) + (512L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr3[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr3 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(50176L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x2 + (50176L*x1) + (150528L*x0))];
                        out_ptr4[static_cast<long>(x1 + (3L*x2) + (150528L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_1 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(64.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3137L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr1 + static_cast<long>(x2), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(3137);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(out_ptr2 + static_cast<long>((-64L) + x2 + (64L*x1) + (200704L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr2 + static_cast<long>(x2), to_float_mask(tmp8));
                            auto tmp14 = tmp12 * tmp13;
                            auto tmp15 = masked_load(in_ptr3 + static_cast<long>(x2), to_float_mask(tmp8));
                            auto tmp16 = tmp14 + tmp15;
                            return tmp16;
                        }
                        ;
                        auto tmp17 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp18 = to_float_mask(tmp4);
                        auto tmp19 = decltype(tmp7)::blendv(tmp17, tmp7, tmp18);
                        tmp19.store(out_ptr3 + static_cast<long>(x2 + (64L*x1) + (200768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_view_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3137L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<int>(x2);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x1 + (64L*x2) + (200768L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(3137);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x1 + (64L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(3136L))) + (200704L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr0 + static_cast<long>(64L + x1 + (64L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(3136L))) + (200768L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        tmp17.store(out_ptr0 + static_cast<long>(x1 + (64L*x2) + (200768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(25096L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(64.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3137L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(64L + x1 + (192L*x2) + (602304L*x0)));
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3137L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(64L + x2 + (192L*x1) + (602304L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.exp();
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (200768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3137L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (64L*x2) + (200768L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3137L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x3 + (8L*x1) + (64L*x2) + (200768L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x3 + (8L*x1) + (64L*x0)));
                            auto tmp2 = tmp0 / tmp1;
                            tmp2.store(out_ptr3 + static_cast<long>(x3 + (8L*x2) + (25096L*x1) + (200768L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3137L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(128L + x3 + (8L*x1) + (192L*x2) + (602304L*x0)));
                            tmp0.store(out_ptr4 + static_cast<long>(x3 + (8L*x2) + (25096L*x1) + (200768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3137L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (8L*x1) + (192L*x2) + (602304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (8L*x2) + (25096L*x1) + (200768L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_view_5 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(16);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (16L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(40);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-16L) + x1 + (24L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(64);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr2[static_cast<long>((-40L) + x1 + (24L*x0))];
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp21 = tmp11 ? tmp14 : tmp20;
                    auto tmp22 = tmp4 ? tmp7 : tmp21;
                    out_ptr0[static_cast<long>(x1 + (64L*x0))] = tmp22;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr3[static_cast<long>((8L*(static_cast<long>(x0) % static_cast<long>(3137L))) + (25096L*(c10::div_floor_integer((x1 + x1_inner), 8L))) + (200768L*(c10::div_floor_integer(x0, 3137L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(8L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp4 = c10::convert<int>((-1L) + (static_cast<long>(x0) % static_cast<long>(3137L)));
                    auto tmp5 = static_cast<int>(0);
                    auto tmp6 = tmp4 >= tmp5;
                    auto tmp7 = [&]
                    {
                        auto tmp8 = masked_load(in_ptr4 + static_cast<long>(x1 + (192L*x0)), to_float_mask(tmp6));
                        auto tmp9 = masked_load(out_ptr0 + static_cast<long>(x1 + (64L*(static_cast<long>(((-1L) + (static_cast<long>(x0) % static_cast<long>(3137L)))) % static_cast<long>(3136L))) + (200704L*(c10::div_floor_integer(x0, 3137L)))), to_float_mask(tmp6));
                        auto tmp10 = tmp8 * tmp9;
                        return tmp10;
                    }
                    ;
                    auto tmp11 = decltype(tmp7())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp7(), to_float_mask(tmp6));
                    auto tmp12 = tmp3 + tmp11;
                    tmp12.store(out_ptr1 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_6 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(64.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12849152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1606144L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_view_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3137L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<int>(x2);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x1 + (64L*x2) + (200768L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(3137);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x1 + (64L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(3136L))) + (200704L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr0 + static_cast<long>(64L + x1 + (64L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(3136L))) + (200768L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        tmp17.store(out_ptr0 + static_cast<long>(x1 + (64L*x2) + (200768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(25096L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(64.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3137L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(64L + x1 + (192L*x2) + (602304L*x0)));
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3137L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(64L + x2 + (192L*x1) + (602304L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.exp();
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (200768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3137L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (64L*x2) + (200768L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3137L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x3 + (8L*x1) + (64L*x2) + (200768L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x3 + (8L*x1) + (64L*x0)));
                            auto tmp2 = tmp0 / tmp1;
                            tmp2.store(out_ptr3 + static_cast<long>(x3 + (8L*x2) + (25096L*x1) + (200768L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3137L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(128L + x3 + (8L*x1) + (192L*x2) + (602304L*x0)));
                            tmp0.store(out_ptr4 + static_cast<long>(x3 + (8L*x2) + (25096L*x1) + (200768L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3137L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (8L*x1) + (192L*x2) + (602304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (8L*x2) + (25096L*x1) + (200768L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_view_12 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(16);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (16L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(40);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-16L) + x1 + (24L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(64);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr2[static_cast<long>((-40L) + x1 + (24L*x0))];
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp21 = tmp11 ? tmp14 : tmp20;
                    auto tmp22 = tmp4 ? tmp7 : tmp21;
                    out_ptr0[static_cast<long>(x1 + (64L*x0))] = tmp22;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr3[static_cast<long>((8L*(static_cast<long>(x0) % static_cast<long>(3137L))) + (25096L*(c10::div_floor_integer((x1 + x1_inner), 8L))) + (200768L*(c10::div_floor_integer(x0, 3137L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(8L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.3535533905932738);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp4 = c10::convert<int>((-1L) + (static_cast<long>(x0) % static_cast<long>(3137L)));
                    auto tmp5 = static_cast<int>(0);
                    auto tmp6 = tmp4 >= tmp5;
                    auto tmp7 = [&]
                    {
                        auto tmp8 = masked_load(in_ptr4 + static_cast<long>(x1 + (192L*x0)), to_float_mask(tmp6));
                        auto tmp9 = masked_load(out_ptr0 + static_cast<long>(x1 + (64L*(static_cast<long>(((-1L) + (static_cast<long>(x0) % static_cast<long>(3137L)))) % static_cast<long>(3136L))) + (200704L*(c10::div_floor_integer(x0, 3137L)))), to_float_mask(tmp6));
                        auto tmp10 = tmp8 * tmp9;
                        return tmp10;
                    }
                    ;
                    auto tmp11 = decltype(tmp7())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp7(), to_float_mask(tmp6));
                    auto tmp12 = tmp3 + tmp11;
                    tmp12.store(out_ptr1 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_13 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25096L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25096L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(64.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12849152L); x0+=static_cast<long>(8L))
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


cpp_fused_clone_15 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(200704L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(64L + x1 + (200768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(64L + x1 + (200768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(64L + x1 + (200768L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (200704L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_16 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(128.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(785L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr1 + static_cast<long>(x2), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(785);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(out_ptr2 + static_cast<long>((-128L) + x2 + (128L*x1) + (100352L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr2 + static_cast<long>(x2), to_float_mask(tmp8));
                            auto tmp14 = tmp12 * tmp13;
                            auto tmp15 = masked_load(in_ptr3 + static_cast<long>(x2), to_float_mask(tmp8));
                            auto tmp16 = tmp14 + tmp15;
                            return tmp16;
                        }
                        ;
                        auto tmp17 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp18 = to_float_mask(tmp4);
                        auto tmp19 = decltype(tmp7)::blendv(tmp17, tmp7, tmp18);
                        tmp19.store(out_ptr3 + static_cast<long>(x2 + (128L*x1) + (100480L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_view_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(785L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<int>(x2);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x1 + (128L*x2) + (100480L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(785);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x1 + (128L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(784L))) + (100352L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr0 + static_cast<long>(128L + x1 + (128L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(784L))) + (100480L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        tmp17.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (100480L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6280L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(6280L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6280L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(785L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(128L + x1 + (384L*x2) + (301440L*x0)));
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(785L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(128L + x2 + (384L*x1) + (301440L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (128L*x0)));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.exp();
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (100480L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(785L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (128L*x2) + (100480L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(785L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x3 + (16L*x1) + (128L*x2) + (100480L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x3 + (16L*x1) + (128L*x0)));
                            auto tmp2 = tmp0 / tmp1;
                            tmp2.store(out_ptr3 + static_cast<long>(x3 + (16L*x2) + (12560L*x1) + (100480L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(785L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(256L + x3 + (16L*x1) + (384L*x2) + (301440L*x0)));
                            tmp0.store(out_ptr4 + static_cast<long>(x3 + (16L*x2) + (12560L*x1) + (100480L*x0)));
                        }
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(785L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (16L*x1) + (384L*x2) + (301440L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (16L*x2) + (12560L*x1) + (100480L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_view_20 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(32);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (32L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(80);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-32L) + x1 + (48L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(128);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr2[static_cast<long>((-80L) + x1 + (48L*x0))];
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp21 = tmp11 ? tmp14 : tmp20;
                    auto tmp22 = tmp4 ? tmp7 : tmp21;
                    out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp22;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6280L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr3[static_cast<long>((16L*(static_cast<long>(x0) % static_cast<long>(785L))) + (12560L*(c10::div_floor_integer((x1 + x1_inner), 16L))) + (100480L*(c10::div_floor_integer(x0, 785L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.25);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp4 = c10::convert<int>((-1L) + (static_cast<long>(x0) % static_cast<long>(785L)));
                    auto tmp5 = static_cast<int>(0);
                    auto tmp6 = tmp4 >= tmp5;
                    auto tmp7 = [&]
                    {
                        auto tmp8 = masked_load(in_ptr4 + static_cast<long>(x1 + (384L*x0)), to_float_mask(tmp6));
                        auto tmp9 = masked_load(out_ptr0 + static_cast<long>(x1 + (128L*(static_cast<long>(((-1L) + (static_cast<long>(x0) % static_cast<long>(785L)))) % static_cast<long>(784L))) + (100352L*(c10::div_floor_integer(x0, 785L)))), to_float_mask(tmp6));
                        auto tmp10 = tmp8 * tmp9;
                        return tmp10;
                    }
                    ;
                    auto tmp11 = decltype(tmp7())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp7(), to_float_mask(tmp6));
                    auto tmp12 = tmp3 + tmp11;
                    tmp12.store(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_21 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6280L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6280L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(128.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6430720L); x0+=static_cast<long>(8L))
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


cpp_fused_add_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(803840L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_view_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(785L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<int>(x2);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x1 + (128L*x2) + (100480L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(785);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x1 + (128L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(784L))) + (100352L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr0 + static_cast<long>(128L + x1 + (128L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(784L))) + (100480L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        tmp17.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (100480L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6280L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(6280L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6280L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(785L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(128L + x1 + (384L*x2) + (301440L*x0)));
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(785L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(128L + x2 + (384L*x1) + (301440L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (128L*x0)));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.exp();
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (100480L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(785L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (128L*x2) + (100480L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(785L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x3 + (16L*x1) + (128L*x2) + (100480L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x3 + (16L*x1) + (128L*x0)));
                            auto tmp2 = tmp0 / tmp1;
                            tmp2.store(out_ptr3 + static_cast<long>(x3 + (16L*x2) + (12560L*x1) + (100480L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(785L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(256L + x3 + (16L*x1) + (384L*x2) + (301440L*x0)));
                            tmp0.store(out_ptr4 + static_cast<long>(x3 + (16L*x2) + (12560L*x1) + (100480L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(785L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (16L*x1) + (384L*x2) + (301440L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (16L*x2) + (12560L*x1) + (100480L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_view_27 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(32);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (32L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(80);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-32L) + x1 + (48L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(128);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr2[static_cast<long>((-80L) + x1 + (48L*x0))];
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp21 = tmp11 ? tmp14 : tmp20;
                    auto tmp22 = tmp4 ? tmp7 : tmp21;
                    out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp22;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6280L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr3[static_cast<long>((16L*(static_cast<long>(x0) % static_cast<long>(785L))) + (12560L*(c10::div_floor_integer((x1 + x1_inner), 16L))) + (100480L*(c10::div_floor_integer(x0, 785L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.25);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp4 = c10::convert<int>((-1L) + (static_cast<long>(x0) % static_cast<long>(785L)));
                    auto tmp5 = static_cast<int>(0);
                    auto tmp6 = tmp4 >= tmp5;
                    auto tmp7 = [&]
                    {
                        auto tmp8 = masked_load(in_ptr4 + static_cast<long>(x1 + (384L*x0)), to_float_mask(tmp6));
                        auto tmp9 = masked_load(out_ptr0 + static_cast<long>(x1 + (128L*(static_cast<long>(((-1L) + (static_cast<long>(x0) % static_cast<long>(785L)))) % static_cast<long>(784L))) + (100352L*(c10::div_floor_integer(x0, 785L)))), to_float_mask(tmp6));
                        auto tmp10 = tmp8 * tmp9;
                        return tmp10;
                    }
                    ;
                    auto tmp11 = decltype(tmp7())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp7(), to_float_mask(tmp6));
                    auto tmp12 = tmp3 + tmp11;
                    tmp12.store(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_28 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6280L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6280L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(128.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6430720L); x0+=static_cast<long>(8L))
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


cpp_fused_clone_30 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(128L + x1 + (100480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x1 + (100480L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(128L + x1 + (100480L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_31 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(320.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(320L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr1 + static_cast<long>(x2), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(out_ptr2 + static_cast<long>((-320L) + x2 + (320L*x1) + (62720L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr2 + static_cast<long>(x2), to_float_mask(tmp8));
                            auto tmp14 = tmp12 * tmp13;
                            auto tmp15 = masked_load(in_ptr3 + static_cast<long>(x2), to_float_mask(tmp8));
                            auto tmp16 = tmp14 + tmp15;
                            return tmp16;
                        }
                        ;
                        auto tmp17 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp18 = to_float_mask(tmp4);
                        auto tmp19 = decltype(tmp7)::blendv(tmp17, tmp7, tmp18);
                        tmp19.store(out_ptr3 + static_cast<long>(x2 + (320L*x1) + (63040L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_view_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<int>(x2);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x1 + (320L*x2) + (63040L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x1 + (320L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(196L))) + (62720L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr0 + static_cast<long>(320L + x1 + (320L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(196L))) + (63040L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        tmp17.store(out_ptr0 + static_cast<long>(x1 + (320L*x2) + (63040L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(320L + x1 + (960L*x2) + (189120L*x0)));
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(320L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(320L + x2 + (960L*x1) + (189120L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (320L*x0)));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.exp();
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (320L*x1) + (63040L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (320L*x2) + (63040L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(40L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x3 + (40L*x1) + (320L*x2) + (63040L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x3 + (40L*x1) + (320L*x0)));
                            auto tmp2 = tmp0 / tmp1;
                            tmp2.store(out_ptr3 + static_cast<long>(x3 + (40L*x2) + (7880L*x1) + (63040L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(40L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(640L + x3 + (40L*x1) + (960L*x2) + (189120L*x0)));
                            tmp0.store(out_ptr4 + static_cast<long>(x3 + (40L*x2) + (7880L*x1) + (63040L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(40L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (40L*x1) + (960L*x2) + (189120L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (40L*x2) + (7880L*x1) + (63040L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_view_35 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(80);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (80L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(200);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-80L) + x1 + (120L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(320);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr2[static_cast<long>((-200L) + x1 + (120L*x0))];
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp21 = tmp11 ? tmp14 : tmp20;
                    auto tmp22 = tmp4 ? tmp7 : tmp21;
                    out_ptr0[static_cast<long>(x1 + (320L*x0))] = tmp22;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr3[static_cast<long>((40L*(static_cast<long>(x0) % static_cast<long>(197L))) + (7880L*(c10::div_floor_integer((x1 + x1_inner), 40L))) + (63040L*(c10::div_floor_integer(x0, 197L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(40L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.15811388300841897);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp4 = c10::convert<int>((-1L) + (static_cast<long>(x0) % static_cast<long>(197L)));
                    auto tmp5 = static_cast<int>(0);
                    auto tmp6 = tmp4 >= tmp5;
                    auto tmp7 = [&]
                    {
                        auto tmp8 = masked_load(in_ptr4 + static_cast<long>(x1 + (960L*x0)), to_float_mask(tmp6));
                        auto tmp9 = masked_load(out_ptr0 + static_cast<long>(x1 + (320L*(static_cast<long>(((-1L) + (static_cast<long>(x0) % static_cast<long>(197L)))) % static_cast<long>(196L))) + (62720L*(c10::div_floor_integer(x0, 197L)))), to_float_mask(tmp6));
                        auto tmp10 = tmp8 * tmp9;
                        return tmp10;
                    }
                    ;
                    auto tmp11 = decltype(tmp7())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp7(), to_float_mask(tmp6));
                    auto tmp12 = tmp3 + tmp11;
                    tmp12.store(out_ptr1 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_36 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(320.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2017280L); x0+=static_cast<long>(8L))
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


cpp_fused_add_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(504320L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_view_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<int>(x2);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x1 + (320L*x2) + (63040L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x1 + (320L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(196L))) + (62720L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr0 + static_cast<long>(320L + x1 + (320L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(196L))) + (63040L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        tmp17.store(out_ptr0 + static_cast<long>(x1 + (320L*x2) + (63040L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(320L + x1 + (960L*x2) + (189120L*x0)));
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(320L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(320L + x2 + (960L*x1) + (189120L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (320L*x0)));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.exp();
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (320L*x1) + (63040L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (320L*x2) + (63040L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(40L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x3 + (40L*x1) + (320L*x2) + (63040L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x3 + (40L*x1) + (320L*x0)));
                            auto tmp2 = tmp0 / tmp1;
                            tmp2.store(out_ptr3 + static_cast<long>(x3 + (40L*x2) + (7880L*x1) + (63040L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(40L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(640L + x3 + (40L*x1) + (960L*x2) + (189120L*x0)));
                            tmp0.store(out_ptr4 + static_cast<long>(x3 + (40L*x2) + (7880L*x1) + (63040L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(40L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (40L*x1) + (960L*x2) + (189120L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (40L*x2) + (7880L*x1) + (63040L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_view_42 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(80);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (80L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(200);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-80L) + x1 + (120L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(320);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr2[static_cast<long>((-200L) + x1 + (120L*x0))];
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp21 = tmp11 ? tmp14 : tmp20;
                    auto tmp22 = tmp4 ? tmp7 : tmp21;
                    out_ptr0[static_cast<long>(x1 + (320L*x0))] = tmp22;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr3[static_cast<long>((40L*(static_cast<long>(x0) % static_cast<long>(197L))) + (7880L*(c10::div_floor_integer((x1 + x1_inner), 40L))) + (63040L*(c10::div_floor_integer(x0, 197L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(40L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.15811388300841897);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp4 = c10::convert<int>((-1L) + (static_cast<long>(x0) % static_cast<long>(197L)));
                    auto tmp5 = static_cast<int>(0);
                    auto tmp6 = tmp4 >= tmp5;
                    auto tmp7 = [&]
                    {
                        auto tmp8 = masked_load(in_ptr4 + static_cast<long>(x1 + (960L*x0)), to_float_mask(tmp6));
                        auto tmp9 = masked_load(out_ptr0 + static_cast<long>(x1 + (320L*(static_cast<long>(((-1L) + (static_cast<long>(x0) % static_cast<long>(197L)))) % static_cast<long>(196L))) + (62720L*(c10::div_floor_integer(x0, 197L)))), to_float_mask(tmp6));
                        auto tmp10 = tmp8 * tmp9;
                        return tmp10;
                    }
                    ;
                    auto tmp11 = decltype(tmp7())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp7(), to_float_mask(tmp6));
                    auto tmp12 = tmp3 + tmp11;
                    tmp12.store(out_ptr1 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_43 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (320L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(320.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (320L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2017280L); x0+=static_cast<long>(8L))
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


cpp_fused_clone_45 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(62720L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(320L + x1 + (63040L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(320L + x1 + (63040L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(320L + x1 + (63040L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (62720L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_46 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(512.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(50L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr1 + static_cast<long>(x2), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(50);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(out_ptr2 + static_cast<long>((-512L) + x2 + (512L*x1) + (25088L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr2 + static_cast<long>(x2), to_float_mask(tmp8));
                            auto tmp14 = tmp12 * tmp13;
                            auto tmp15 = masked_load(in_ptr3 + static_cast<long>(x2), to_float_mask(tmp8));
                            auto tmp16 = tmp14 + tmp15;
                            return tmp16;
                        }
                        ;
                        auto tmp17 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp18 = to_float_mask(tmp4);
                        auto tmp19 = decltype(tmp7)::blendv(tmp17, tmp7, tmp18);
                        tmp19.store(out_ptr3 + static_cast<long>(x2 + (512L*x1) + (25600L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_view_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(50L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<int>(x2);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x1 + (512L*x2) + (25600L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(50);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x1 + (512L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(49L))) + (25088L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr0 + static_cast<long>(512L + x1 + (512L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(49L))) + (25600L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        tmp17.store(out_ptr0 + static_cast<long>(x1 + (512L*x2) + (25600L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(50L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (76800L*x0)));
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(50L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x2 + (1536L*x1) + (76800L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.exp();
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (512L*x1) + (25600L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(50L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (512L*x2) + (25600L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(50L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (25600L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x3 + (64L*x1) + (512L*x0)));
                            auto tmp2 = tmp0 / tmp1;
                            tmp2.store(out_ptr3 + static_cast<long>(x3 + (64L*x2) + (3200L*x1) + (25600L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(50L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(1024L + x3 + (64L*x1) + (1536L*x2) + (76800L*x0)));
                            tmp0.store(out_ptr4 + static_cast<long>(x3 + (64L*x2) + (3200L*x1) + (25600L*x0)));
                        }
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(50L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (1536L*x2) + (76800L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (3200L*x1) + (25600L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_view_50 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(128);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(320);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-128L) + x1 + (192L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(512);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr2[static_cast<long>((-320L) + x1 + (192L*x0))];
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp21 = tmp11 ? tmp14 : tmp20;
                    auto tmp22 = tmp4 ? tmp7 : tmp21;
                    out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp22;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr3[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(50L))) + (3200L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (25600L*(c10::div_floor_integer(x0, 50L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.125);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp4 = c10::convert<int>((-1L) + (static_cast<long>(x0) % static_cast<long>(50L)));
                    auto tmp5 = static_cast<int>(0);
                    auto tmp6 = tmp4 >= tmp5;
                    auto tmp7 = [&]
                    {
                        auto tmp8 = masked_load(in_ptr4 + static_cast<long>(x1 + (1536L*x0)), to_float_mask(tmp6));
                        auto tmp9 = masked_load(out_ptr0 + static_cast<long>(x1 + (512L*(static_cast<long>(((-1L) + (static_cast<long>(x0) % static_cast<long>(50L)))) % static_cast<long>(49L))) + (25088L*(c10::div_floor_integer(x0, 50L)))), to_float_mask(tmp6));
                        auto tmp10 = tmp8 * tmp9;
                        return tmp10;
                    }
                    ;
                    auto tmp11 = decltype(tmp7())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp7(), to_float_mask(tmp6));
                    auto tmp12 = tmp3 + tmp11;
                    tmp12.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_51 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(512.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(819200L); x0+=static_cast<long>(8L))
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


cpp_fused_add_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(204800L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_view_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(50L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<int>(x2);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x1 + (512L*x2) + (25600L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(50);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x1 + (512L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(49L))) + (25088L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr0 + static_cast<long>(512L + x1 + (512L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(49L))) + (25600L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        tmp17.store(out_ptr0 + static_cast<long>(x1 + (512L*x2) + (25600L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(50L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (76800L*x0)));
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(50L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x2 + (1536L*x1) + (76800L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.exp();
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (512L*x1) + (25600L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(50L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (512L*x2) + (25600L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(50L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (25600L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x3 + (64L*x1) + (512L*x0)));
                            auto tmp2 = tmp0 / tmp1;
                            tmp2.store(out_ptr3 + static_cast<long>(x3 + (64L*x2) + (3200L*x1) + (25600L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(50L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(1024L + x3 + (64L*x1) + (1536L*x2) + (76800L*x0)));
                            tmp0.store(out_ptr4 + static_cast<long>(x3 + (64L*x2) + (3200L*x1) + (25600L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(50L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (1536L*x2) + (76800L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (3200L*x1) + (25600L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_view_57 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(128);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(320);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-128L) + x1 + (192L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(512);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr2[static_cast<long>((-320L) + x1 + (192L*x0))];
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp21 = tmp11 ? tmp14 : tmp20;
                    auto tmp22 = tmp4 ? tmp7 : tmp21;
                    out_ptr0[static_cast<long>(x1 + (512L*x0))] = tmp22;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr3[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(50L))) + (3200L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (25600L*(c10::div_floor_integer(x0, 50L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(0.125);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp4 = c10::convert<int>((-1L) + (static_cast<long>(x0) % static_cast<long>(50L)));
                    auto tmp5 = static_cast<int>(0);
                    auto tmp6 = tmp4 >= tmp5;
                    auto tmp7 = [&]
                    {
                        auto tmp8 = masked_load(in_ptr4 + static_cast<long>(x1 + (1536L*x0)), to_float_mask(tmp6));
                        auto tmp9 = masked_load(out_ptr0 + static_cast<long>(x1 + (512L*(static_cast<long>(((-1L) + (static_cast<long>(x0) % static_cast<long>(50L)))) % static_cast<long>(49L))) + (25088L*(c10::div_floor_integer(x0, 50L)))), to_float_mask(tmp6));
                        auto tmp10 = tmp8 * tmp9;
                        return tmp10;
                    }
                    ;
                    auto tmp11 = decltype(tmp7())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp7(), to_float_mask(tmp6));
                    auto tmp12 = tmp3 + tmp11;
                    tmp12.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_58 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(512.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(819200L); x0+=static_cast<long>(8L))
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


cpp_fused_add_clone_native_layer_norm_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(512.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (25600L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp4.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_detach_native_layer_norm_native_layer_norm_backward_61 = async_compile.cpp('''
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
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3200L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (3200L*x1) + (3200L*x1_inner) + (25600L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (8L*x2) + (25600L*x0)), static_cast<long>(8L));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3200L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (3200L*x1) + (3200L*x1_inner) + (25600L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (8L*x2) + (25600L*x0)), static_cast<long>(8L));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7880L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (7880L*x1) + (7880L*x1_inner) + (63040L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (8L*x2) + (63040L*x0)), static_cast<long>(8L));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr5 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7880L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (7880L*x1) + (7880L*x1_inner) + (63040L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (8L*x2) + (63040L*x0)), static_cast<long>(8L));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr6 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(320.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(6280L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr7 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr7 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12560L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (12560L*x1) + (12560L*x1_inner) + (100480L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr4 + static_cast<long>(x1 + (8L*x2) + (100480L*x0)), static_cast<long>(8L));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(6280L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr8 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr8 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12560L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (12560L*x1) + (12560L*x1_inner) + (100480L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr5 + static_cast<long>(x1 + (8L*x2) + (100480L*x0)), static_cast<long>(8L));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr9 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(25096L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr10 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(64.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr10 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(25096L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (25096L*x1) + (25096L*x1_inner) + (200768L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr6 + static_cast<long>(x1 + (8L*x2) + (200768L*x0)), static_cast<long>(8L));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(25096L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr11 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(64.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr11 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(25096L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2 + (25096L*x1) + (25096L*x1_inner) + (200768L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr7 + static_cast<long>(x1 + (8L*x2) + (200768L*x0)), static_cast<long>(8L));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr12 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(64.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr12 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153 = args
    args.clear()
    assert_size_stride(primals_1, (1, 1, 64), (64, 64, 1))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (64, ), (1, ))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_10, (1, 1, 128), (128, 128, 1))
    assert_size_stride(primals_11, (128, ), (1, ))
    assert_size_stride(primals_12, (128, ), (1, ))
    assert_size_stride(primals_13, (128, ), (1, ))
    assert_size_stride(primals_14, (128, ), (1, ))
    assert_size_stride(primals_15, (128, ), (1, ))
    assert_size_stride(primals_16, (128, ), (1, ))
    assert_size_stride(primals_17, (128, ), (1, ))
    assert_size_stride(primals_18, (128, ), (1, ))
    assert_size_stride(primals_19, (1, 1, 320), (320, 320, 1))
    assert_size_stride(primals_20, (320, ), (1, ))
    assert_size_stride(primals_21, (320, ), (1, ))
    assert_size_stride(primals_22, (320, ), (1, ))
    assert_size_stride(primals_23, (320, ), (1, ))
    assert_size_stride(primals_24, (320, ), (1, ))
    assert_size_stride(primals_25, (320, ), (1, ))
    assert_size_stride(primals_26, (320, ), (1, ))
    assert_size_stride(primals_27, (320, ), (1, ))
    assert_size_stride(primals_28, (1, 1, 512), (512, 512, 1))
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
    assert_size_stride(primals_39, (64, 3, 4, 4), (48, 16, 4, 1))
    assert_size_stride(primals_40, (64, ), (1, ))
    assert_size_stride(primals_41, (64, ), (1, ))
    assert_size_stride(primals_42, (64, ), (1, ))
    assert_size_stride(primals_43, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_44, (64, ), (1, ))
    assert_size_stride(primals_45, (192, 64), (64, 1))
    assert_size_stride(primals_46, (192, ), (1, ))
    assert_size_stride(primals_47, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_48, (16, ), (1, ))
    assert_size_stride(primals_49, (24, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_50, (24, ), (1, ))
    assert_size_stride(primals_51, (24, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_52, (24, ), (1, ))
    assert_size_stride(primals_53, (64, 64), (64, 1))
    assert_size_stride(primals_54, (64, ), (1, ))
    assert_size_stride(primals_55, (512, 64), (64, 1))
    assert_size_stride(primals_56, (512, ), (1, ))
    assert_size_stride(primals_57, (64, 512), (512, 1))
    assert_size_stride(primals_58, (64, ), (1, ))
    assert_size_stride(primals_59, (192, 64), (64, 1))
    assert_size_stride(primals_60, (192, ), (1, ))
    assert_size_stride(primals_61, (64, 64), (64, 1))
    assert_size_stride(primals_62, (64, ), (1, ))
    assert_size_stride(primals_63, (512, 64), (64, 1))
    assert_size_stride(primals_64, (512, ), (1, ))
    assert_size_stride(primals_65, (64, 512), (512, 1))
    assert_size_stride(primals_66, (64, ), (1, ))
    assert_size_stride(primals_67, (128, 64, 2, 2), (256, 4, 2, 1))
    assert_size_stride(primals_68, (128, ), (1, ))
    assert_size_stride(primals_69, (128, ), (1, ))
    assert_size_stride(primals_70, (128, ), (1, ))
    assert_size_stride(primals_71, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_72, (128, ), (1, ))
    assert_size_stride(primals_73, (384, 128), (128, 1))
    assert_size_stride(primals_74, (384, ), (1, ))
    assert_size_stride(primals_75, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_76, (32, ), (1, ))
    assert_size_stride(primals_77, (48, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_78, (48, ), (1, ))
    assert_size_stride(primals_79, (48, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_80, (48, ), (1, ))
    assert_size_stride(primals_81, (128, 128), (128, 1))
    assert_size_stride(primals_82, (128, ), (1, ))
    assert_size_stride(primals_83, (1024, 128), (128, 1))
    assert_size_stride(primals_84, (1024, ), (1, ))
    assert_size_stride(primals_85, (128, 1024), (1024, 1))
    assert_size_stride(primals_86, (128, ), (1, ))
    assert_size_stride(primals_87, (384, 128), (128, 1))
    assert_size_stride(primals_88, (384, ), (1, ))
    assert_size_stride(primals_89, (128, 128), (128, 1))
    assert_size_stride(primals_90, (128, ), (1, ))
    assert_size_stride(primals_91, (1024, 128), (128, 1))
    assert_size_stride(primals_92, (1024, ), (1, ))
    assert_size_stride(primals_93, (128, 1024), (1024, 1))
    assert_size_stride(primals_94, (128, ), (1, ))
    assert_size_stride(primals_95, (320, 128, 2, 2), (512, 4, 2, 1))
    assert_size_stride(primals_96, (320, ), (1, ))
    assert_size_stride(primals_97, (320, ), (1, ))
    assert_size_stride(primals_98, (320, ), (1, ))
    assert_size_stride(primals_99, (320, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_100, (320, ), (1, ))
    assert_size_stride(primals_101, (960, 320), (320, 1))
    assert_size_stride(primals_102, (960, ), (1, ))
    assert_size_stride(primals_103, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_104, (80, ), (1, ))
    assert_size_stride(primals_105, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_106, (120, ), (1, ))
    assert_size_stride(primals_107, (120, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_108, (120, ), (1, ))
    assert_size_stride(primals_109, (320, 320), (320, 1))
    assert_size_stride(primals_110, (320, ), (1, ))
    assert_size_stride(primals_111, (1280, 320), (320, 1))
    assert_size_stride(primals_112, (1280, ), (1, ))
    assert_size_stride(primals_113, (320, 1280), (1280, 1))
    assert_size_stride(primals_114, (320, ), (1, ))
    assert_size_stride(primals_115, (960, 320), (320, 1))
    assert_size_stride(primals_116, (960, ), (1, ))
    assert_size_stride(primals_117, (320, 320), (320, 1))
    assert_size_stride(primals_118, (320, ), (1, ))
    assert_size_stride(primals_119, (1280, 320), (320, 1))
    assert_size_stride(primals_120, (1280, ), (1, ))
    assert_size_stride(primals_121, (320, 1280), (1280, 1))
    assert_size_stride(primals_122, (320, ), (1, ))
    assert_size_stride(primals_123, (512, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(primals_124, (512, ), (1, ))
    assert_size_stride(primals_125, (512, ), (1, ))
    assert_size_stride(primals_126, (512, ), (1, ))
    assert_size_stride(primals_127, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_128, (512, ), (1, ))
    assert_size_stride(primals_129, (1536, 512), (512, 1))
    assert_size_stride(primals_130, (1536, ), (1, ))
    assert_size_stride(primals_131, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_132, (128, ), (1, ))
    assert_size_stride(primals_133, (192, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_134, (192, ), (1, ))
    assert_size_stride(primals_135, (192, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_136, (192, ), (1, ))
    assert_size_stride(primals_137, (512, 512), (512, 1))
    assert_size_stride(primals_138, (512, ), (1, ))
    assert_size_stride(primals_139, (2048, 512), (512, 1))
    assert_size_stride(primals_140, (2048, ), (1, ))
    assert_size_stride(primals_141, (512, 2048), (2048, 1))
    assert_size_stride(primals_142, (512, ), (1, ))
    assert_size_stride(primals_143, (1536, 512), (512, 1))
    assert_size_stride(primals_144, (1536, ), (1, ))
    assert_size_stride(primals_145, (512, 512), (512, 1))
    assert_size_stride(primals_146, (512, ), (1, ))
    assert_size_stride(primals_147, (2048, 512), (512, 1))
    assert_size_stride(primals_148, (2048, ), (1, ))
    assert_size_stride(primals_149, (512, 2048), (2048, 1))
    assert_size_stride(primals_150, (512, ), (1, ))
    assert_size_stride(primals_151, (1000, 512), (512, 1))
    assert_size_stride(primals_152, (1000, ), (1, ))
    assert_size_stride(primals_153, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((64, 3, 4, 4), (48, 1, 12, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((128, 64, 2, 2), (256, 1, 128, 64), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((320, 128, 2, 2), (512, 1, 256, 128), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((512, 320, 2, 2), (1280, 1, 640, 320), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    cpp_fused_0(c_void_p(primals_39.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(primals_123.data_ptr()), c_void_p(primals_153.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()))
    del primals_123
    del primals_153
    del primals_39
    del primals_67
    del primals_95
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf5 = extern_kernels.convolution(buf4, buf0, primals_40, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf5, (8, 64, 56, 56), (200704, 1, 3584, 64))
    del primals_40
    buf6 = empty_strided((8, 3136, 1), (3136, 1, 25088), device='cpu', dtype=torch.float32)
    buf7 = empty_strided((8, 3136, 1), (3136, 1, 25088), device='cpu', dtype=torch.float32)
    buf9 = empty((8, 3136, 64), device='cpu', dtype=torch.float32)
    buf10 = empty((8, 3137, 64), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_1(c_void_p(buf5.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()))
    del buf6
    del primals_1
    del primals_42
    # Source Nodes: [l__mod___serial_blocks1_0_cpe_proj], Original ATen: [aten.convolution]
    buf11 = extern_kernels.convolution(reinterpret_tensor(buf10, (8, 64, 56, 56), (200768, 1, 3584, 64), 64), primals_43, primals_44, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64)
    assert_size_stride(buf11, (8, 64, 56, 56), (200704, 1, 3584, 64))
    buf12 = empty((8, 3137, 64), device='cpu', dtype=torch.float32)
    buf13 = empty((8, 3137, 1), device='cpu', dtype=torch.float32)
    buf14 = empty_strided((8, 3137, 1), (3137, 1, 25096), device='cpu', dtype=torch.float32)
    buf16 = reinterpret_tensor(buf14, (8, 3137, 1), (3137, 1, 1), 0); del buf14  # reuse
    buf17 = empty((25096, 64), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_view_2(c_void_p(buf16.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf17.data_ptr()))
    del primals_3
    buf18 = empty((25096, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___serial_blocks1_0_factoratt_crpe_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_46, buf17, reinterpret_tensor(primals_45, (64, 192), (1, 64), 0), alpha=1, beta=1, out=buf18)
    del primals_46
    buf19 = empty_strided((8, 8, 1, 8), (64, 8, 512, 1), device='cpu', dtype=torch.float32)
    buf20 = empty_strided((8, 8, 3137, 8), (200768, 8, 64, 1), device='cpu', dtype=torch.float32)
    buf21 = empty_strided((8, 8, 1, 8), (64, 8, 512, 1), device='cpu', dtype=torch.float32)
    buf22 = empty((8, 8, 3137, 8), device='cpu', dtype=torch.float32)
    buf23 = empty((8, 8, 3137, 8), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_3(c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf23.data_ptr()))
    buf24 = empty((64, 8, 8), device='cpu', dtype=torch.float32)
    # Source Nodes: [factor_att], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf22, (64, 8, 3137), (25096, 1, 8), 0), reinterpret_tensor(buf23, (64, 3137, 8), (25096, 8, 1), 0), out=buf24)
    buf25 = reinterpret_tensor(buf20, (8, 8, 3137, 8), (200768, 25096, 8, 1), 0); del buf20  # reuse
    cpp_fused_clone_4(c_void_p(buf18.data_ptr()), c_void_p(buf25.data_ptr()))
    buf26 = empty((64, 3137, 8), device='cpu', dtype=torch.float32)
    # Source Nodes: [factor_att_1], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf25, (64, 3137, 8), (25096, 8, 1), 0), reinterpret_tensor(buf24, (64, 8, 8), (64, 8, 1), 0), out=buf26)
    # Source Nodes: [l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_0], Original ATen: [aten.convolution]
    buf27 = extern_kernels.convolution(reinterpret_tensor(buf18, (8, 16, 56, 56), (602304, 1, 10752, 192), 320), primals_47, primals_48, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16)
    assert_size_stride(buf27, (8, 16, 56, 56), (50176, 1, 896, 16))
    # Source Nodes: [l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_1], Original ATen: [aten.convolution]
    buf28 = extern_kernels.convolution(reinterpret_tensor(buf18, (8, 24, 56, 56), (602304, 1, 10752, 192), 336), primals_49, primals_50, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24)
    assert_size_stride(buf28, (8, 24, 56, 56), (75264, 1, 1344, 24))
    # Source Nodes: [l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_2], Original ATen: [aten.convolution]
    buf29 = extern_kernels.convolution(reinterpret_tensor(buf18, (8, 24, 56, 56), (602304, 1, 10752, 192), 360), primals_51, primals_52, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24)
    assert_size_stride(buf29, (8, 24, 56, 56), (75264, 1, 1344, 24))
    buf30 = buf11; del buf11  # reuse
    buf31 = empty((25096, 64), device='cpu', dtype=torch.float32)
    cpp_fused_cat_view_5(c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()))
    del buf27
    del buf28
    del buf29
    buf32 = reinterpret_tensor(buf26, (25096, 64), (64, 1), 0); del buf26  # reuse
    # Source Nodes: [x_11], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_54, buf31, reinterpret_tensor(primals_53, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf32)
    del primals_54
    buf33 = empty_strided((8, 3137, 1), (3137, 1, 25096), device='cpu', dtype=torch.float32)
    buf34 = empty_strided((8, 3137, 1), (3137, 1, 25096), device='cpu', dtype=torch.float32)
    buf36 = empty((8, 3137, 64), device='cpu', dtype=torch.float32)
    buf37 = empty((25096, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_6(c_void_p(buf12.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()))
    del primals_5
    buf38 = empty((25096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_15], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_56, buf37, reinterpret_tensor(primals_55, (64, 512), (1, 64), 0), alpha=1, beta=1, out=buf38)
    del primals_56
    buf39 = empty((25096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_7(c_void_p(buf38.data_ptr()), c_void_p(buf39.data_ptr()))
    buf40 = empty((25096, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_19], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_58, buf39, reinterpret_tensor(primals_57, (512, 64), (1, 512), 0), alpha=1, beta=1, out=buf40)
    del primals_58
    buf41 = reinterpret_tensor(buf40, (8, 3137, 64), (200768, 64, 1), 0); del buf40  # reuse
    cpp_fused_add_8(c_void_p(buf41.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf32.data_ptr()))
    # Source Nodes: [l__mod___serial_blocks1_0_cpe_proj_1], Original ATen: [aten.convolution]
    buf42 = extern_kernels.convolution(reinterpret_tensor(buf41, (8, 64, 56, 56), (200768, 1, 3584, 64), 64), primals_43, primals_44, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64)
    assert_size_stride(buf42, (8, 64, 56, 56), (200704, 1, 3584, 64))
    del primals_44
    buf43 = reinterpret_tensor(buf32, (8, 3137, 64), (200768, 64, 1), 0); del buf32  # reuse
    buf44 = reinterpret_tensor(buf33, (8, 3137, 1), (3137, 1, 1), 0); del buf33  # reuse
    buf45 = empty_strided((8, 3137, 1), (3137, 1, 25096), device='cpu', dtype=torch.float32)
    buf47 = reinterpret_tensor(buf45, (8, 3137, 1), (3137, 1, 1), 0); del buf45  # reuse
    buf48 = empty((25096, 64), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_view_9(c_void_p(buf47.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf48.data_ptr()))
    del primals_7
    buf49 = empty((25096, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___serial_blocks1_1_factoratt_crpe_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_60, buf48, reinterpret_tensor(primals_59, (64, 192), (1, 64), 0), alpha=1, beta=1, out=buf49)
    del primals_60
    buf50 = buf21; del buf21  # reuse
    buf51 = empty_strided((8, 8, 3137, 8), (200768, 8, 64, 1), device='cpu', dtype=torch.float32)
    buf52 = buf19; del buf19  # reuse
    buf53 = empty((8, 8, 3137, 8), device='cpu', dtype=torch.float32)
    buf54 = empty((8, 8, 3137, 8), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_10(c_void_p(buf49.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf54.data_ptr()))
    del buf50
    del buf52
    buf55 = empty((64, 8, 8), device='cpu', dtype=torch.float32)
    # Source Nodes: [factor_att_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf53, (64, 8, 3137), (25096, 1, 8), 0), reinterpret_tensor(buf54, (64, 3137, 8), (25096, 8, 1), 0), out=buf55)
    buf56 = reinterpret_tensor(buf51, (8, 8, 3137, 8), (200768, 25096, 8, 1), 0); del buf51  # reuse
    cpp_fused_clone_11(c_void_p(buf49.data_ptr()), c_void_p(buf56.data_ptr()))
    buf57 = empty((64, 3137, 8), device='cpu', dtype=torch.float32)
    # Source Nodes: [factor_att_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf56, (64, 3137, 8), (25096, 8, 1), 0), reinterpret_tensor(buf55, (64, 8, 8), (64, 8, 1), 0), out=buf57)
    # Source Nodes: [l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_3], Original ATen: [aten.convolution]
    buf58 = extern_kernels.convolution(reinterpret_tensor(buf49, (8, 16, 56, 56), (602304, 1, 10752, 192), 320), primals_47, primals_48, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16)
    assert_size_stride(buf58, (8, 16, 56, 56), (50176, 1, 896, 16))
    del primals_48
    # Source Nodes: [l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_4], Original ATen: [aten.convolution]
    buf59 = extern_kernels.convolution(reinterpret_tensor(buf49, (8, 24, 56, 56), (602304, 1, 10752, 192), 336), primals_49, primals_50, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24)
    assert_size_stride(buf59, (8, 24, 56, 56), (75264, 1, 1344, 24))
    del primals_50
    # Source Nodes: [l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_5], Original ATen: [aten.convolution]
    buf60 = extern_kernels.convolution(reinterpret_tensor(buf49, (8, 24, 56, 56), (602304, 1, 10752, 192), 360), primals_51, primals_52, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24)
    assert_size_stride(buf60, (8, 24, 56, 56), (75264, 1, 1344, 24))
    del primals_52
    buf61 = buf42; del buf42  # reuse
    buf62 = empty((25096, 64), device='cpu', dtype=torch.float32)
    cpp_fused_cat_view_12(c_void_p(buf58.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()))
    del buf58
    del buf59
    del buf60
    buf63 = reinterpret_tensor(buf57, (25096, 64), (64, 1), 0); del buf57  # reuse
    # Source Nodes: [x_29], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_62, buf62, reinterpret_tensor(primals_61, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf63)
    del primals_62
    buf64 = empty_strided((8, 3137, 1), (3137, 1, 25096), device='cpu', dtype=torch.float32)
    buf65 = empty_strided((8, 3137, 1), (3137, 1, 25096), device='cpu', dtype=torch.float32)
    buf67 = empty((8, 3137, 64), device='cpu', dtype=torch.float32)
    buf68 = empty((25096, 64), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_13(c_void_p(buf43.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf68.data_ptr()))
    del buf64
    del primals_9
    buf69 = empty((25096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_33], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_64, buf68, reinterpret_tensor(primals_63, (64, 512), (1, 64), 0), alpha=1, beta=1, out=buf69)
    del primals_64
    buf70 = empty((25096, 512), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_14(c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()))
    buf71 = empty((25096, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_37], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_66, buf70, reinterpret_tensor(primals_65, (512, 64), (1, 512), 0), alpha=1, beta=1, out=buf71)
    del primals_66
    buf72 = buf5; del buf5  # reuse
    cpp_fused_clone_15(c_void_p(buf43.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf72.data_ptr()))
    # Source Nodes: [x_40], Original ATen: [aten.convolution]
    buf73 = extern_kernels.convolution(buf72, buf1, primals_68, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf73, (8, 128, 28, 28), (100352, 1, 3584, 128))
    del primals_68
    buf74 = empty_strided((8, 784, 1), (784, 1, 6272), device='cpu', dtype=torch.float32)
    buf75 = empty_strided((8, 784, 1), (784, 1, 6272), device='cpu', dtype=torch.float32)
    buf77 = empty((8, 784, 128), device='cpu', dtype=torch.float32)
    buf78 = empty((8, 785, 128), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_16(c_void_p(buf73.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(primals_70.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf78.data_ptr()))
    del buf74
    del primals_10
    del primals_70
    # Source Nodes: [l__mod___serial_blocks2_0_cpe_proj], Original ATen: [aten.convolution]
    buf79 = extern_kernels.convolution(reinterpret_tensor(buf78, (8, 128, 28, 28), (100480, 1, 3584, 128), 128), primals_71, primals_72, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128)
    assert_size_stride(buf79, (8, 128, 28, 28), (100352, 1, 3584, 128))
    buf80 = empty((8, 785, 128), device='cpu', dtype=torch.float32)
    buf81 = empty((8, 785, 1), device='cpu', dtype=torch.float32)
    buf82 = empty_strided((8, 785, 1), (785, 1, 6280), device='cpu', dtype=torch.float32)
    buf84 = reinterpret_tensor(buf82, (8, 785, 1), (785, 1, 1), 0); del buf82  # reuse
    buf85 = empty((6280, 128), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_view_17(c_void_p(buf84.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf85.data_ptr()))
    del primals_12
    buf86 = empty((6280, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___serial_blocks2_0_factoratt_crpe_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_74, buf85, reinterpret_tensor(primals_73, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf86)
    del primals_74
    buf87 = empty_strided((8, 8, 1, 16), (128, 16, 1024, 1), device='cpu', dtype=torch.float32)
    buf88 = empty_strided((8, 8, 785, 16), (100480, 16, 128, 1), device='cpu', dtype=torch.float32)
    buf89 = empty_strided((8, 8, 1, 16), (128, 16, 1024, 1), device='cpu', dtype=torch.float32)
    buf90 = empty((8, 8, 785, 16), device='cpu', dtype=torch.float32)
    buf91 = empty((8, 8, 785, 16), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_18(c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf91.data_ptr()))
    buf92 = empty((64, 16, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [factor_att_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf90, (64, 16, 785), (12560, 1, 16), 0), reinterpret_tensor(buf91, (64, 785, 16), (12560, 16, 1), 0), out=buf92)
    buf93 = reinterpret_tensor(buf88, (8, 8, 785, 16), (100480, 12560, 16, 1), 0); del buf88  # reuse
    cpp_fused_clone_19(c_void_p(buf86.data_ptr()), c_void_p(buf93.data_ptr()))
    buf94 = empty((64, 785, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [factor_att_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf93, (64, 785, 16), (12560, 16, 1), 0), reinterpret_tensor(buf92, (64, 16, 16), (256, 16, 1), 0), out=buf94)
    # Source Nodes: [l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_0], Original ATen: [aten.convolution]
    buf95 = extern_kernels.convolution(reinterpret_tensor(buf86, (8, 32, 28, 28), (301440, 1, 10752, 384), 640), primals_75, primals_76, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32)
    assert_size_stride(buf95, (8, 32, 28, 28), (25088, 1, 896, 32))
    # Source Nodes: [l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_1], Original ATen: [aten.convolution]
    buf96 = extern_kernels.convolution(reinterpret_tensor(buf86, (8, 48, 28, 28), (301440, 1, 10752, 384), 672), primals_77, primals_78, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48)
    assert_size_stride(buf96, (8, 48, 28, 28), (37632, 1, 1344, 48))
    # Source Nodes: [l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_2], Original ATen: [aten.convolution]
    buf97 = extern_kernels.convolution(reinterpret_tensor(buf86, (8, 48, 28, 28), (301440, 1, 10752, 384), 720), primals_79, primals_80, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48)
    assert_size_stride(buf97, (8, 48, 28, 28), (37632, 1, 1344, 48))
    buf98 = buf79; del buf79  # reuse
    buf99 = empty((6280, 128), device='cpu', dtype=torch.float32)
    cpp_fused_cat_view_20(c_void_p(buf95.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf99.data_ptr()))
    del buf95
    del buf96
    del buf97
    buf100 = reinterpret_tensor(buf94, (6280, 128), (128, 1), 0); del buf94  # reuse
    # Source Nodes: [x_51], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_82, buf99, reinterpret_tensor(primals_81, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf100)
    del primals_82
    buf101 = empty_strided((8, 785, 1), (785, 1, 6280), device='cpu', dtype=torch.float32)
    buf102 = empty_strided((8, 785, 1), (785, 1, 6280), device='cpu', dtype=torch.float32)
    buf104 = empty((8, 785, 128), device='cpu', dtype=torch.float32)
    buf105 = empty((6280, 128), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_21(c_void_p(buf80.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf105.data_ptr()))
    del primals_14
    buf106 = empty((6280, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_55], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_84, buf105, reinterpret_tensor(primals_83, (128, 1024), (1, 128), 0), alpha=1, beta=1, out=buf106)
    del primals_84
    buf107 = empty((6280, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_22(c_void_p(buf106.data_ptr()), c_void_p(buf107.data_ptr()))
    buf108 = empty((6280, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_59], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_86, buf107, reinterpret_tensor(primals_85, (1024, 128), (1, 1024), 0), alpha=1, beta=1, out=buf108)
    del primals_86
    buf109 = reinterpret_tensor(buf108, (8, 785, 128), (100480, 128, 1), 0); del buf108  # reuse
    cpp_fused_add_23(c_void_p(buf109.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf100.data_ptr()))
    # Source Nodes: [l__mod___serial_blocks2_0_cpe_proj_1], Original ATen: [aten.convolution]
    buf110 = extern_kernels.convolution(reinterpret_tensor(buf109, (8, 128, 28, 28), (100480, 1, 3584, 128), 128), primals_71, primals_72, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128)
    assert_size_stride(buf110, (8, 128, 28, 28), (100352, 1, 3584, 128))
    del primals_72
    buf111 = reinterpret_tensor(buf100, (8, 785, 128), (100480, 128, 1), 0); del buf100  # reuse
    buf112 = reinterpret_tensor(buf101, (8, 785, 1), (785, 1, 1), 0); del buf101  # reuse
    buf113 = empty_strided((8, 785, 1), (785, 1, 6280), device='cpu', dtype=torch.float32)
    buf115 = reinterpret_tensor(buf113, (8, 785, 1), (785, 1, 1), 0); del buf113  # reuse
    buf116 = empty((6280, 128), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_view_24(c_void_p(buf115.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf116.data_ptr()))
    del primals_16
    buf117 = empty((6280, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___serial_blocks2_1_factoratt_crpe_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_88, buf116, reinterpret_tensor(primals_87, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf117)
    del primals_88
    buf118 = buf89; del buf89  # reuse
    buf119 = empty_strided((8, 8, 785, 16), (100480, 16, 128, 1), device='cpu', dtype=torch.float32)
    buf120 = buf87; del buf87  # reuse
    buf121 = empty((8, 8, 785, 16), device='cpu', dtype=torch.float32)
    buf122 = empty((8, 8, 785, 16), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_25(c_void_p(buf117.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf122.data_ptr()))
    del buf118
    del buf120
    buf123 = empty((64, 16, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [factor_att_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf121, (64, 16, 785), (12560, 1, 16), 0), reinterpret_tensor(buf122, (64, 785, 16), (12560, 16, 1), 0), out=buf123)
    buf124 = reinterpret_tensor(buf119, (8, 8, 785, 16), (100480, 12560, 16, 1), 0); del buf119  # reuse
    cpp_fused_clone_26(c_void_p(buf117.data_ptr()), c_void_p(buf124.data_ptr()))
    buf125 = empty((64, 785, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [factor_att_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf124, (64, 785, 16), (12560, 16, 1), 0), reinterpret_tensor(buf123, (64, 16, 16), (256, 16, 1), 0), out=buf125)
    # Source Nodes: [l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_3], Original ATen: [aten.convolution]
    buf126 = extern_kernels.convolution(reinterpret_tensor(buf117, (8, 32, 28, 28), (301440, 1, 10752, 384), 640), primals_75, primals_76, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32)
    assert_size_stride(buf126, (8, 32, 28, 28), (25088, 1, 896, 32))
    del primals_76
    # Source Nodes: [l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_4], Original ATen: [aten.convolution]
    buf127 = extern_kernels.convolution(reinterpret_tensor(buf117, (8, 48, 28, 28), (301440, 1, 10752, 384), 672), primals_77, primals_78, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48)
    assert_size_stride(buf127, (8, 48, 28, 28), (37632, 1, 1344, 48))
    del primals_78
    # Source Nodes: [l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_5], Original ATen: [aten.convolution]
    buf128 = extern_kernels.convolution(reinterpret_tensor(buf117, (8, 48, 28, 28), (301440, 1, 10752, 384), 720), primals_79, primals_80, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48)
    assert_size_stride(buf128, (8, 48, 28, 28), (37632, 1, 1344, 48))
    del primals_80
    buf129 = buf110; del buf110  # reuse
    buf130 = empty((6280, 128), device='cpu', dtype=torch.float32)
    cpp_fused_cat_view_27(c_void_p(buf126.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf130.data_ptr()))
    del buf127
    del buf128
    buf131 = reinterpret_tensor(buf125, (6280, 128), (128, 1), 0); del buf125  # reuse
    # Source Nodes: [x_69], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_90, buf130, reinterpret_tensor(primals_89, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf131)
    del primals_90
    buf132 = empty_strided((8, 785, 1), (785, 1, 6280), device='cpu', dtype=torch.float32)
    buf133 = empty_strided((8, 785, 1), (785, 1, 6280), device='cpu', dtype=torch.float32)
    buf135 = empty((8, 785, 128), device='cpu', dtype=torch.float32)
    buf136 = empty((6280, 128), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_28(c_void_p(buf111.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf136.data_ptr()))
    del buf132
    del primals_18
    buf137 = empty((6280, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_73], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_92, buf136, reinterpret_tensor(primals_91, (128, 1024), (1, 128), 0), alpha=1, beta=1, out=buf137)
    del primals_92
    buf138 = empty((6280, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_29(c_void_p(buf137.data_ptr()), c_void_p(buf138.data_ptr()))
    buf139 = empty((6280, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_77], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_94, buf138, reinterpret_tensor(primals_93, (1024, 128), (1, 1024), 0), alpha=1, beta=1, out=buf139)
    del primals_94
    buf140 = buf73; del buf73  # reuse
    cpp_fused_clone_30(c_void_p(buf111.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf140.data_ptr()))
    # Source Nodes: [x_80], Original ATen: [aten.convolution]
    buf141 = extern_kernels.convolution(buf140, buf2, primals_96, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf141, (8, 320, 14, 14), (62720, 1, 4480, 320))
    del primals_96
    buf142 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf143 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf145 = empty((8, 196, 320), device='cpu', dtype=torch.float32)
    buf146 = empty((8, 197, 320), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_31(c_void_p(buf141.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(primals_97.data_ptr()), c_void_p(primals_98.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf146.data_ptr()))
    del buf142
    del primals_19
    del primals_98
    # Source Nodes: [l__mod___serial_blocks3_0_cpe_proj], Original ATen: [aten.convolution]
    buf147 = extern_kernels.convolution(reinterpret_tensor(buf146, (8, 320, 14, 14), (63040, 1, 4480, 320), 320), primals_99, primals_100, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=320)
    assert_size_stride(buf147, (8, 320, 14, 14), (62720, 1, 4480, 320))
    buf148 = empty((8, 197, 320), device='cpu', dtype=torch.float32)
    buf149 = empty((8, 197, 1), device='cpu', dtype=torch.float32)
    buf150 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf152 = reinterpret_tensor(buf150, (8, 197, 1), (197, 1, 1), 0); del buf150  # reuse
    buf153 = empty((1576, 320), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_view_32(c_void_p(buf152.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf153.data_ptr()))
    del primals_21
    buf154 = empty((1576, 960), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___serial_blocks3_0_factoratt_crpe_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_102, buf153, reinterpret_tensor(primals_101, (320, 960), (1, 320), 0), alpha=1, beta=1, out=buf154)
    del primals_102
    buf155 = empty_strided((8, 8, 1, 40), (320, 40, 2560, 1), device='cpu', dtype=torch.float32)
    buf156 = empty_strided((8, 8, 197, 40), (63040, 40, 320, 1), device='cpu', dtype=torch.float32)
    buf157 = empty_strided((8, 8, 1, 40), (320, 40, 2560, 1), device='cpu', dtype=torch.float32)
    buf158 = empty((8, 8, 197, 40), device='cpu', dtype=torch.float32)
    buf159 = empty((8, 8, 197, 40), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_33(c_void_p(buf154.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf159.data_ptr()))
    buf160 = empty((64, 40, 40), device='cpu', dtype=torch.float32)
    # Source Nodes: [factor_att_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf158, (64, 40, 197), (7880, 1, 40), 0), reinterpret_tensor(buf159, (64, 197, 40), (7880, 40, 1), 0), out=buf160)
    buf161 = reinterpret_tensor(buf156, (8, 8, 197, 40), (63040, 7880, 40, 1), 0); del buf156  # reuse
    cpp_fused_clone_34(c_void_p(buf154.data_ptr()), c_void_p(buf161.data_ptr()))
    buf162 = empty((64, 197, 40), device='cpu', dtype=torch.float32)
    # Source Nodes: [factor_att_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf161, (64, 197, 40), (7880, 40, 1), 0), reinterpret_tensor(buf160, (64, 40, 40), (1600, 40, 1), 0), out=buf162)
    # Source Nodes: [l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_0], Original ATen: [aten.convolution]
    buf163 = extern_kernels.convolution(reinterpret_tensor(buf154, (8, 80, 14, 14), (189120, 1, 13440, 960), 1600), primals_103, primals_104, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80)
    assert_size_stride(buf163, (8, 80, 14, 14), (15680, 1, 1120, 80))
    # Source Nodes: [l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_1], Original ATen: [aten.convolution]
    buf164 = extern_kernels.convolution(reinterpret_tensor(buf154, (8, 120, 14, 14), (189120, 1, 13440, 960), 1680), primals_105, primals_106, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120)
    assert_size_stride(buf164, (8, 120, 14, 14), (23520, 1, 1680, 120))
    # Source Nodes: [l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_2], Original ATen: [aten.convolution]
    buf165 = extern_kernels.convolution(reinterpret_tensor(buf154, (8, 120, 14, 14), (189120, 1, 13440, 960), 1800), primals_107, primals_108, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120)
    assert_size_stride(buf165, (8, 120, 14, 14), (23520, 1, 1680, 120))
    buf166 = buf147; del buf147  # reuse
    buf167 = empty((1576, 320), device='cpu', dtype=torch.float32)
    cpp_fused_cat_view_35(c_void_p(buf163.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf167.data_ptr()))
    del buf163
    del buf164
    del buf165
    buf168 = reinterpret_tensor(buf162, (1576, 320), (320, 1), 0); del buf162  # reuse
    # Source Nodes: [x_91], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_110, buf167, reinterpret_tensor(primals_109, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf168)
    del primals_110
    buf169 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf170 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf172 = empty((8, 197, 320), device='cpu', dtype=torch.float32)
    buf173 = empty((1576, 320), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_36(c_void_p(buf148.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf173.data_ptr()))
    del primals_23
    buf174 = empty((1576, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_95], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_112, buf173, reinterpret_tensor(primals_111, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf174)
    del primals_112
    buf175 = empty((1576, 1280), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_37(c_void_p(buf174.data_ptr()), c_void_p(buf175.data_ptr()))
    buf176 = empty((1576, 320), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_99], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_114, buf175, reinterpret_tensor(primals_113, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf176)
    del primals_114
    buf177 = reinterpret_tensor(buf176, (8, 197, 320), (63040, 320, 1), 0); del buf176  # reuse
    cpp_fused_add_38(c_void_p(buf177.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf168.data_ptr()))
    # Source Nodes: [l__mod___serial_blocks3_0_cpe_proj_1], Original ATen: [aten.convolution]
    buf178 = extern_kernels.convolution(reinterpret_tensor(buf177, (8, 320, 14, 14), (63040, 1, 4480, 320), 320), primals_99, primals_100, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=320)
    assert_size_stride(buf178, (8, 320, 14, 14), (62720, 1, 4480, 320))
    del primals_100
    buf179 = reinterpret_tensor(buf168, (8, 197, 320), (63040, 320, 1), 0); del buf168  # reuse
    buf180 = reinterpret_tensor(buf169, (8, 197, 1), (197, 1, 1), 0); del buf169  # reuse
    buf181 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf183 = reinterpret_tensor(buf181, (8, 197, 1), (197, 1, 1), 0); del buf181  # reuse
    buf184 = empty((1576, 320), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_view_39(c_void_p(buf183.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf184.data_ptr()))
    del primals_25
    buf185 = empty((1576, 960), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___serial_blocks3_1_factoratt_crpe_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_116, buf184, reinterpret_tensor(primals_115, (320, 960), (1, 320), 0), alpha=1, beta=1, out=buf185)
    del primals_116
    buf186 = buf157; del buf157  # reuse
    buf187 = empty_strided((8, 8, 197, 40), (63040, 40, 320, 1), device='cpu', dtype=torch.float32)
    buf188 = buf155; del buf155  # reuse
    buf189 = empty((8, 8, 197, 40), device='cpu', dtype=torch.float32)
    buf190 = empty((8, 8, 197, 40), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_40(c_void_p(buf185.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf190.data_ptr()))
    del buf186
    del buf188
    buf191 = empty((64, 40, 40), device='cpu', dtype=torch.float32)
    # Source Nodes: [factor_att_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf189, (64, 40, 197), (7880, 1, 40), 0), reinterpret_tensor(buf190, (64, 197, 40), (7880, 40, 1), 0), out=buf191)
    buf192 = reinterpret_tensor(buf187, (8, 8, 197, 40), (63040, 7880, 40, 1), 0); del buf187  # reuse
    cpp_fused_clone_41(c_void_p(buf185.data_ptr()), c_void_p(buf192.data_ptr()))
    buf193 = empty((64, 197, 40), device='cpu', dtype=torch.float32)
    # Source Nodes: [factor_att_11], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf192, (64, 197, 40), (7880, 40, 1), 0), reinterpret_tensor(buf191, (64, 40, 40), (1600, 40, 1), 0), out=buf193)
    # Source Nodes: [l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_3], Original ATen: [aten.convolution]
    buf194 = extern_kernels.convolution(reinterpret_tensor(buf185, (8, 80, 14, 14), (189120, 1, 13440, 960), 1600), primals_103, primals_104, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80)
    assert_size_stride(buf194, (8, 80, 14, 14), (15680, 1, 1120, 80))
    del primals_104
    # Source Nodes: [l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_4], Original ATen: [aten.convolution]
    buf195 = extern_kernels.convolution(reinterpret_tensor(buf185, (8, 120, 14, 14), (189120, 1, 13440, 960), 1680), primals_105, primals_106, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120)
    assert_size_stride(buf195, (8, 120, 14, 14), (23520, 1, 1680, 120))
    del primals_106
    # Source Nodes: [l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_5], Original ATen: [aten.convolution]
    buf196 = extern_kernels.convolution(reinterpret_tensor(buf185, (8, 120, 14, 14), (189120, 1, 13440, 960), 1800), primals_107, primals_108, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120)
    assert_size_stride(buf196, (8, 120, 14, 14), (23520, 1, 1680, 120))
    del primals_108
    buf197 = buf178; del buf178  # reuse
    buf198 = empty((1576, 320), device='cpu', dtype=torch.float32)
    cpp_fused_cat_view_42(c_void_p(buf194.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf198.data_ptr()))
    del buf194
    del buf195
    del buf196
    buf199 = reinterpret_tensor(buf193, (1576, 320), (320, 1), 0); del buf193  # reuse
    # Source Nodes: [x_109], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_118, buf198, reinterpret_tensor(primals_117, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf199)
    del primals_118
    buf200 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf201 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf203 = empty((8, 197, 320), device='cpu', dtype=torch.float32)
    buf204 = empty((1576, 320), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_43(c_void_p(buf179.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf204.data_ptr()))
    del buf200
    del primals_27
    buf205 = empty((1576, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_113], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_120, buf204, reinterpret_tensor(primals_119, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf205)
    del primals_120
    buf206 = empty((1576, 1280), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_44(c_void_p(buf205.data_ptr()), c_void_p(buf206.data_ptr()))
    buf207 = empty((1576, 320), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_117], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_122, buf206, reinterpret_tensor(primals_121, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf207)
    del primals_122
    buf208 = buf141; del buf141  # reuse
    cpp_fused_clone_45(c_void_p(buf179.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()))
    # Source Nodes: [x_120], Original ATen: [aten.convolution]
    buf209 = extern_kernels.convolution(buf208, buf3, primals_124, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf209, (8, 512, 7, 7), (25088, 1, 3584, 512))
    del primals_124
    buf210 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf211 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf213 = reinterpret_tensor(buf126, (8, 49, 512), (25088, 512, 1), 0); del buf126  # reuse
    buf214 = empty((8, 50, 512), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_46(c_void_p(buf209.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(primals_126.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf214.data_ptr()))
    del buf209
    del buf210
    del primals_126
    del primals_28
    # Source Nodes: [l__mod___serial_blocks4_0_cpe_proj], Original ATen: [aten.convolution]
    buf215 = extern_kernels.convolution(reinterpret_tensor(buf214, (8, 512, 7, 7), (25600, 1, 3584, 512), 512), primals_127, primals_128, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512)
    assert_size_stride(buf215, (8, 512, 7, 7), (25088, 1, 3584, 512))
    buf216 = empty((8, 50, 512), device='cpu', dtype=torch.float32)
    buf217 = empty((8, 50, 1), device='cpu', dtype=torch.float32)
    buf218 = empty_strided((8, 50, 1), (50, 1, 400), device='cpu', dtype=torch.float32)
    buf220 = reinterpret_tensor(buf218, (8, 50, 1), (50, 1, 1), 0); del buf218  # reuse
    buf221 = empty((400, 512), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_view_47(c_void_p(buf220.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf221.data_ptr()))
    del primals_30
    buf222 = empty((400, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___serial_blocks4_0_factoratt_crpe_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_130, buf221, reinterpret_tensor(primals_129, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf222)
    del primals_130
    buf223 = empty_strided((8, 8, 1, 64), (512, 64, 4096, 1), device='cpu', dtype=torch.float32)
    buf224 = empty_strided((8, 8, 50, 64), (25600, 64, 512, 1), device='cpu', dtype=torch.float32)
    buf225 = empty_strided((8, 8, 1, 64), (512, 64, 4096, 1), device='cpu', dtype=torch.float32)
    buf226 = empty((8, 8, 50, 64), device='cpu', dtype=torch.float32)
    buf227 = empty((8, 8, 50, 64), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_48(c_void_p(buf222.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf227.data_ptr()))
    buf228 = empty((64, 64, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [factor_att_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf226, (64, 64, 50), (3200, 1, 64), 0), reinterpret_tensor(buf227, (64, 50, 64), (3200, 64, 1), 0), out=buf228)
    buf229 = reinterpret_tensor(buf224, (8, 8, 50, 64), (25600, 3200, 64, 1), 0); del buf224  # reuse
    cpp_fused_clone_49(c_void_p(buf222.data_ptr()), c_void_p(buf229.data_ptr()))
    buf230 = empty((64, 50, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [factor_att_13], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf229, (64, 50, 64), (3200, 64, 1), 0), reinterpret_tensor(buf228, (64, 64, 64), (4096, 64, 1), 0), out=buf230)
    # Source Nodes: [l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_0], Original ATen: [aten.convolution]
    buf231 = extern_kernels.convolution(reinterpret_tensor(buf222, (8, 128, 7, 7), (76800, 1, 10752, 1536), 2560), primals_131, primals_132, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128)
    assert_size_stride(buf231, (8, 128, 7, 7), (6272, 1, 896, 128))
    # Source Nodes: [l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_1], Original ATen: [aten.convolution]
    buf232 = extern_kernels.convolution(reinterpret_tensor(buf222, (8, 192, 7, 7), (76800, 1, 10752, 1536), 2688), primals_133, primals_134, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192)
    assert_size_stride(buf232, (8, 192, 7, 7), (9408, 1, 1344, 192))
    # Source Nodes: [l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_2], Original ATen: [aten.convolution]
    buf233 = extern_kernels.convolution(reinterpret_tensor(buf222, (8, 192, 7, 7), (76800, 1, 10752, 1536), 2880), primals_135, primals_136, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192)
    assert_size_stride(buf233, (8, 192, 7, 7), (9408, 1, 1344, 192))
    buf234 = buf215; del buf215  # reuse
    buf235 = empty((400, 512), device='cpu', dtype=torch.float32)
    cpp_fused_cat_view_50(c_void_p(buf231.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf235.data_ptr()))
    del buf231
    del buf232
    del buf233
    buf236 = reinterpret_tensor(buf230, (400, 512), (512, 1), 0); del buf230  # reuse
    # Source Nodes: [x_131], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_138, buf235, reinterpret_tensor(primals_137, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf236)
    del primals_138
    buf237 = empty_strided((8, 50, 1), (50, 1, 400), device='cpu', dtype=torch.float32)
    buf238 = empty_strided((8, 50, 1), (50, 1, 400), device='cpu', dtype=torch.float32)
    buf240 = empty((8, 50, 512), device='cpu', dtype=torch.float32)
    buf241 = empty((400, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_51(c_void_p(buf216.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf241.data_ptr()))
    del primals_32
    buf242 = empty((400, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_135], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_140, buf241, reinterpret_tensor(primals_139, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf242)
    del primals_140
    buf243 = empty((400, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_52(c_void_p(buf242.data_ptr()), c_void_p(buf243.data_ptr()))
    buf244 = empty((400, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_139], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_142, buf243, reinterpret_tensor(primals_141, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf244)
    del primals_142
    buf245 = reinterpret_tensor(buf244, (8, 50, 512), (25600, 512, 1), 0); del buf244  # reuse
    cpp_fused_add_53(c_void_p(buf245.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf236.data_ptr()))
    # Source Nodes: [l__mod___serial_blocks4_0_cpe_proj_1], Original ATen: [aten.convolution]
    buf246 = extern_kernels.convolution(reinterpret_tensor(buf245, (8, 512, 7, 7), (25600, 1, 3584, 512), 512), primals_127, primals_128, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512)
    assert_size_stride(buf246, (8, 512, 7, 7), (25088, 1, 3584, 512))
    del primals_128
    buf247 = reinterpret_tensor(buf236, (8, 50, 512), (25600, 512, 1), 0); del buf236  # reuse
    buf248 = reinterpret_tensor(buf237, (8, 50, 1), (50, 1, 1), 0); del buf237  # reuse
    buf249 = empty_strided((8, 50, 1), (50, 1, 400), device='cpu', dtype=torch.float32)
    buf251 = reinterpret_tensor(buf249, (8, 50, 1), (50, 1, 1), 0); del buf249  # reuse
    buf252 = empty((400, 512), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_view_54(c_void_p(buf251.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(primals_34.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf252.data_ptr()))
    del primals_34
    buf253 = empty((400, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___serial_blocks4_1_factoratt_crpe_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_144, buf252, reinterpret_tensor(primals_143, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf253)
    del primals_144
    buf254 = buf225; del buf225  # reuse
    buf255 = empty_strided((8, 8, 50, 64), (25600, 64, 512, 1), device='cpu', dtype=torch.float32)
    buf256 = buf223; del buf223  # reuse
    buf257 = empty((8, 8, 50, 64), device='cpu', dtype=torch.float32)
    buf258 = empty((8, 8, 50, 64), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_55(c_void_p(buf253.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf258.data_ptr()))
    del buf254
    buf259 = empty((64, 64, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [factor_att_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf257, (64, 64, 50), (3200, 1, 64), 0), reinterpret_tensor(buf258, (64, 50, 64), (3200, 64, 1), 0), out=buf259)
    buf260 = reinterpret_tensor(buf255, (8, 8, 50, 64), (25600, 3200, 64, 1), 0); del buf255  # reuse
    cpp_fused_clone_56(c_void_p(buf253.data_ptr()), c_void_p(buf260.data_ptr()))
    buf261 = empty((64, 50, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [factor_att_15], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf260, (64, 50, 64), (3200, 64, 1), 0), reinterpret_tensor(buf259, (64, 64, 64), (4096, 64, 1), 0), out=buf261)
    # Source Nodes: [l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_3], Original ATen: [aten.convolution]
    buf262 = extern_kernels.convolution(reinterpret_tensor(buf253, (8, 128, 7, 7), (76800, 1, 10752, 1536), 2560), primals_131, primals_132, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128)
    assert_size_stride(buf262, (8, 128, 7, 7), (6272, 1, 896, 128))
    del primals_132
    # Source Nodes: [l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_4], Original ATen: [aten.convolution]
    buf263 = extern_kernels.convolution(reinterpret_tensor(buf253, (8, 192, 7, 7), (76800, 1, 10752, 1536), 2688), primals_133, primals_134, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192)
    assert_size_stride(buf263, (8, 192, 7, 7), (9408, 1, 1344, 192))
    del primals_134
    # Source Nodes: [l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_5], Original ATen: [aten.convolution]
    buf264 = extern_kernels.convolution(reinterpret_tensor(buf253, (8, 192, 7, 7), (76800, 1, 10752, 1536), 2880), primals_135, primals_136, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192)
    assert_size_stride(buf264, (8, 192, 7, 7), (9408, 1, 1344, 192))
    del primals_136
    buf265 = buf246; del buf246  # reuse
    buf266 = empty((400, 512), device='cpu', dtype=torch.float32)
    cpp_fused_cat_view_57(c_void_p(buf262.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf266.data_ptr()))
    del buf262
    del buf263
    del buf264
    buf267 = reinterpret_tensor(buf261, (400, 512), (512, 1), 0); del buf261  # reuse
    # Source Nodes: [x_149], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_146, buf266, reinterpret_tensor(primals_145, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf267)
    del primals_146
    buf268 = empty_strided((8, 50, 1), (50, 1, 400), device='cpu', dtype=torch.float32)
    buf269 = empty_strided((8, 50, 1), (50, 1, 400), device='cpu', dtype=torch.float32)
    buf271 = empty((8, 50, 512), device='cpu', dtype=torch.float32)
    buf272 = empty((400, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_58(c_void_p(buf247.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf272.data_ptr()))
    del primals_36
    buf273 = empty((400, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_153], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_148, buf272, reinterpret_tensor(primals_147, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf273)
    del primals_148
    buf274 = empty((400, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_59(c_void_p(buf273.data_ptr()), c_void_p(buf274.data_ptr()))
    buf275 = empty((400, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_157], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_150, buf274, reinterpret_tensor(primals_149, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf275)
    del primals_150
    buf276 = buf268; del buf268  # reuse
    buf277 = empty_strided((8, 50, 1), (50, 1, 400), device='cpu', dtype=torch.float32)
    buf279 = empty((8, 50, 512), device='cpu', dtype=torch.float32)
    buf280 = reinterpret_tensor(buf256, (8, 512), (512, 1), 0); del buf256  # reuse
    cpp_fused_add_clone_native_layer_norm_60(c_void_p(buf247.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf280.data_ptr()))
    del buf276
    del primals_38
    buf281 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [pred], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_152, buf280, reinterpret_tensor(primals_151, (512, 1000), (1, 512), 0), alpha=1, beta=1, out=buf281)
    del primals_152
    buf282 = reinterpret_tensor(buf277, (8, 50, 1), (50, 1, 1), 0); del buf277  # reuse
    buf283 = reinterpret_tensor(buf269, (8, 50, 1), (50, 1, 1), 0); del buf269  # reuse
    buf284 = reinterpret_tensor(buf275, (8, 8, 50, 64), (25600, 1, 512, 8), 0); del buf275  # reuse
    buf285 = reinterpret_tensor(buf238, (8, 50, 1), (50, 1, 1), 0); del buf238  # reuse
    buf286 = reinterpret_tensor(buf267, (8, 8, 50, 64), (25600, 1, 512, 8), 0); del buf267  # reuse
    buf287 = reinterpret_tensor(buf211, (8, 49, 1), (49, 1, 1), 0); del buf211  # reuse
    buf288 = reinterpret_tensor(buf201, (8, 197, 1), (197, 1, 1), 0); del buf201  # reuse
    buf289 = reinterpret_tensor(buf207, (8, 8, 197, 40), (63040, 1, 320, 8), 0); del buf207  # reuse
    buf290 = reinterpret_tensor(buf170, (8, 197, 1), (197, 1, 1), 0); del buf170  # reuse
    buf291 = reinterpret_tensor(buf199, (8, 8, 197, 40), (63040, 1, 320, 8), 0); del buf199  # reuse
    buf292 = reinterpret_tensor(buf143, (8, 196, 1), (196, 1, 1), 0); del buf143  # reuse
    buf293 = reinterpret_tensor(buf133, (8, 785, 1), (785, 1, 1), 0); del buf133  # reuse
    buf294 = reinterpret_tensor(buf139, (8, 8, 785, 16), (100480, 1, 128, 8), 0); del buf139  # reuse
    buf295 = reinterpret_tensor(buf102, (8, 785, 1), (785, 1, 1), 0); del buf102  # reuse
    buf296 = reinterpret_tensor(buf131, (8, 8, 785, 16), (100480, 1, 128, 8), 0); del buf131  # reuse
    buf297 = reinterpret_tensor(buf75, (8, 784, 1), (784, 1, 1), 0); del buf75  # reuse
    buf298 = reinterpret_tensor(buf65, (8, 3137, 1), (3137, 1, 1), 0); del buf65  # reuse
    buf299 = reinterpret_tensor(buf71, (8, 8, 3137, 8), (200768, 1, 64, 8), 0); del buf71  # reuse
    buf300 = reinterpret_tensor(buf34, (8, 3137, 1), (3137, 1, 1), 0); del buf34  # reuse
    buf301 = reinterpret_tensor(buf63, (8, 8, 3137, 8), (200768, 1, 64, 8), 0); del buf63  # reuse
    buf302 = reinterpret_tensor(buf7, (8, 3136, 1), (3136, 1, 1), 0); del buf7  # reuse
    cpp_fused_add_detach_native_layer_norm_native_layer_norm_backward_61(c_void_p(buf282.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf301.data_ptr()))
    return (buf281, primals_2, primals_4, primals_6, primals_8, primals_11, primals_13, primals_15, primals_17, primals_20, primals_22, primals_24, primals_26, primals_29, primals_31, primals_33, primals_35, primals_37, buf0, primals_41, primals_43, primals_47, primals_49, primals_51, buf1, primals_69, primals_71, primals_75, primals_77, primals_79, buf2, primals_97, primals_99, primals_103, primals_105, primals_107, buf3, primals_125, primals_127, primals_131, primals_133, primals_135, buf4, buf9, reinterpret_tensor(buf10, (8, 64, 56, 56), (200768, 1, 3584, 64), 64), buf12, buf13, buf16, buf17, reinterpret_tensor(buf18, (8, 8, 3136, 8), (602304, 8, 192, 1), 192), reinterpret_tensor(buf18, (8, 16, 56, 56), (602304, 1, 10752, 192), 320), reinterpret_tensor(buf18, (8, 24, 56, 56), (602304, 1, 10752, 192), 336), reinterpret_tensor(buf18, (8, 24, 56, 56), (602304, 1, 10752, 192), 360), buf30, buf31, buf36, buf37, buf38, buf39, reinterpret_tensor(buf41, (8, 64, 56, 56), (200768, 1, 3584, 64), 64), buf43, buf44, buf47, buf48, reinterpret_tensor(buf49, (8, 8, 3136, 8), (602304, 8, 192, 1), 192), reinterpret_tensor(buf49, (8, 16, 56, 56), (602304, 1, 10752, 192), 320), reinterpret_tensor(buf49, (8, 24, 56, 56), (602304, 1, 10752, 192), 336), reinterpret_tensor(buf49, (8, 24, 56, 56), (602304, 1, 10752, 192), 360), buf61, buf62, buf67, buf68, buf69, buf70, buf72, buf77, reinterpret_tensor(buf78, (8, 128, 28, 28), (100480, 1, 3584, 128), 128), buf80, buf81, buf84, buf85, reinterpret_tensor(buf86, (8, 8, 784, 16), (301440, 16, 384, 1), 384), reinterpret_tensor(buf86, (8, 32, 28, 28), (301440, 1, 10752, 384), 640), reinterpret_tensor(buf86, (8, 48, 28, 28), (301440, 1, 10752, 384), 672), reinterpret_tensor(buf86, (8, 48, 28, 28), (301440, 1, 10752, 384), 720), buf98, buf99, buf104, buf105, buf106, buf107, reinterpret_tensor(buf109, (8, 128, 28, 28), (100480, 1, 3584, 128), 128), buf111, buf112, buf115, buf116, reinterpret_tensor(buf117, (8, 8, 784, 16), (301440, 16, 384, 1), 384), reinterpret_tensor(buf117, (8, 32, 28, 28), (301440, 1, 10752, 384), 640), reinterpret_tensor(buf117, (8, 48, 28, 28), (301440, 1, 10752, 384), 672), reinterpret_tensor(buf117, (8, 48, 28, 28), (301440, 1, 10752, 384), 720), buf129, buf130, buf135, buf136, buf137, buf138, buf140, buf145, reinterpret_tensor(buf146, (8, 320, 14, 14), (63040, 1, 4480, 320), 320), buf148, buf149, buf152, buf153, reinterpret_tensor(buf154, (8, 8, 196, 40), (189120, 40, 960, 1), 960), reinterpret_tensor(buf154, (8, 80, 14, 14), (189120, 1, 13440, 960), 1600), reinterpret_tensor(buf154, (8, 120, 14, 14), (189120, 1, 13440, 960), 1680), reinterpret_tensor(buf154, (8, 120, 14, 14), (189120, 1, 13440, 960), 1800), buf166, buf167, buf172, buf173, buf174, buf175, reinterpret_tensor(buf177, (8, 320, 14, 14), (63040, 1, 4480, 320), 320), buf179, buf180, buf183, buf184, reinterpret_tensor(buf185, (8, 8, 196, 40), (189120, 40, 960, 1), 960), reinterpret_tensor(buf185, (8, 80, 14, 14), (189120, 1, 13440, 960), 1600), reinterpret_tensor(buf185, (8, 120, 14, 14), (189120, 1, 13440, 960), 1680), reinterpret_tensor(buf185, (8, 120, 14, 14), (189120, 1, 13440, 960), 1800), buf197, buf198, buf203, buf204, buf205, buf206, buf208, buf213, reinterpret_tensor(buf214, (8, 512, 7, 7), (25600, 1, 3584, 512), 512), buf216, buf217, buf220, buf221, reinterpret_tensor(buf222, (8, 8, 49, 64), (76800, 64, 1536, 1), 1536), reinterpret_tensor(buf222, (8, 128, 7, 7), (76800, 1, 10752, 1536), 2560), reinterpret_tensor(buf222, (8, 192, 7, 7), (76800, 1, 10752, 1536), 2688), reinterpret_tensor(buf222, (8, 192, 7, 7), (76800, 1, 10752, 1536), 2880), buf234, buf235, buf240, buf241, buf242, buf243, reinterpret_tensor(buf245, (8, 512, 7, 7), (25600, 1, 3584, 512), 512), buf247, buf248, buf251, buf252, reinterpret_tensor(buf253, (8, 8, 49, 64), (76800, 64, 1536, 1), 1536), reinterpret_tensor(buf253, (8, 128, 7, 7), (76800, 1, 10752, 1536), 2560), reinterpret_tensor(buf253, (8, 192, 7, 7), (76800, 1, 10752, 1536), 2688), reinterpret_tensor(buf253, (8, 192, 7, 7), (76800, 1, 10752, 1536), 2880), buf265, buf266, buf271, buf272, buf273, buf274, buf279, buf280, reinterpret_tensor(primals_151, (1000, 512), (512, 1), 0), buf282, reinterpret_tensor(primals_149, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_147, (2048, 512), (512, 1), 0), buf283, reinterpret_tensor(primals_145, (512, 512), (512, 1), 0), reinterpret_tensor(buf260, (64, 64, 50), (3200, 1, 64), 0), reinterpret_tensor(buf259, (64, 64, 64), (4096, 1, 64), 0), reinterpret_tensor(buf257, (64, 50, 64), (3200, 64, 1), 0), reinterpret_tensor(buf258, (64, 64, 50), (3200, 1, 64), 0), buf284, reinterpret_tensor(primals_143, (1536, 512), (512, 1), 0), reinterpret_tensor(primals_141, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_139, (2048, 512), (512, 1), 0), buf285, reinterpret_tensor(primals_137, (512, 512), (512, 1), 0), reinterpret_tensor(buf229, (64, 64, 50), (3200, 1, 64), 0), reinterpret_tensor(buf228, (64, 64, 64), (4096, 1, 64), 0), reinterpret_tensor(buf226, (64, 50, 64), (3200, 64, 1), 0), reinterpret_tensor(buf227, (64, 64, 50), (3200, 1, 64), 0), buf286, reinterpret_tensor(primals_129, (1536, 512), (512, 1), 0), buf287, reinterpret_tensor(primals_121, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_119, (1280, 320), (320, 1), 0), buf288, reinterpret_tensor(primals_117, (320, 320), (320, 1), 0), reinterpret_tensor(buf192, (64, 40, 197), (7880, 1, 40), 0), reinterpret_tensor(buf191, (64, 40, 40), (1600, 1, 40), 0), reinterpret_tensor(buf189, (64, 197, 40), (7880, 40, 1), 0), reinterpret_tensor(buf190, (64, 40, 197), (7880, 1, 40), 0), buf289, reinterpret_tensor(primals_115, (960, 320), (320, 1), 0), reinterpret_tensor(primals_113, (320, 1280), (1280, 1), 0), reinterpret_tensor(primals_111, (1280, 320), (320, 1), 0), buf290, reinterpret_tensor(primals_109, (320, 320), (320, 1), 0), reinterpret_tensor(buf161, (64, 40, 197), (7880, 1, 40), 0), reinterpret_tensor(buf160, (64, 40, 40), (1600, 1, 40), 0), reinterpret_tensor(buf158, (64, 197, 40), (7880, 40, 1), 0), reinterpret_tensor(buf159, (64, 40, 197), (7880, 1, 40), 0), buf291, reinterpret_tensor(primals_101, (960, 320), (320, 1), 0), buf292, reinterpret_tensor(primals_93, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_91, (1024, 128), (128, 1), 0), buf293, reinterpret_tensor(primals_89, (128, 128), (128, 1), 0), reinterpret_tensor(buf124, (64, 16, 785), (12560, 1, 16), 0), reinterpret_tensor(buf123, (64, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf121, (64, 785, 16), (12560, 16, 1), 0), reinterpret_tensor(buf122, (64, 16, 785), (12560, 1, 16), 0), buf294, reinterpret_tensor(primals_87, (384, 128), (128, 1), 0), reinterpret_tensor(primals_85, (128, 1024), (1024, 1), 0), reinterpret_tensor(primals_83, (1024, 128), (128, 1), 0), buf295, reinterpret_tensor(primals_81, (128, 128), (128, 1), 0), reinterpret_tensor(buf93, (64, 16, 785), (12560, 1, 16), 0), reinterpret_tensor(buf92, (64, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf90, (64, 785, 16), (12560, 16, 1), 0), reinterpret_tensor(buf91, (64, 16, 785), (12560, 1, 16), 0), buf296, reinterpret_tensor(primals_73, (384, 128), (128, 1), 0), buf297, reinterpret_tensor(primals_65, (64, 512), (512, 1), 0), reinterpret_tensor(primals_63, (512, 64), (64, 1), 0), buf298, reinterpret_tensor(primals_61, (64, 64), (64, 1), 0), reinterpret_tensor(buf56, (64, 8, 3137), (25096, 1, 8), 0), reinterpret_tensor(buf55, (64, 8, 8), (64, 1, 8), 0), reinterpret_tensor(buf53, (64, 3137, 8), (25096, 8, 1), 0), reinterpret_tensor(buf54, (64, 8, 3137), (25096, 1, 8), 0), buf299, reinterpret_tensor(primals_59, (192, 64), (64, 1), 0), reinterpret_tensor(primals_57, (64, 512), (512, 1), 0), reinterpret_tensor(primals_55, (512, 64), (64, 1), 0), buf300, reinterpret_tensor(primals_53, (64, 64), (64, 1), 0), reinterpret_tensor(buf25, (64, 8, 3137), (25096, 1, 8), 0), reinterpret_tensor(buf24, (64, 8, 8), (64, 1, 8), 0), reinterpret_tensor(buf22, (64, 3137, 8), (25096, 8, 1), 0), reinterpret_tensor(buf23, (64, 8, 3137), (25096, 1, 8), 0), buf301, reinterpret_tensor(primals_45, (192, 64), (64, 1), 0), buf302, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 1, 64), (64, 64, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((1, 1, 128), (128, 128, 1), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((1, 1, 320), (320, 320, 1), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((1, 1, 512), (512, 512, 1), device='cpu', dtype=torch.float32)
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
    primals_39 = rand_strided((64, 3, 4, 4), (48, 16, 4, 1), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((192, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((24, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((24, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((64, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((512, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((64, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((192, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((64, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((512, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((64, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((128, 64, 2, 2), (256, 4, 2, 1), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((384, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((48, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((48, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((1024, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((384, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((1024, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((320, 128, 2, 2), (512, 4, 2, 1), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((320, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((960, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((120, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((960, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((512, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((192, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((192, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((1000, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('coat_lite_mini', benchmark_compiled_module)
