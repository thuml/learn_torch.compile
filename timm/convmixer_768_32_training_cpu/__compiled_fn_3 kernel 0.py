
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
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x2 + (49L*x1) + (147L*x0))];
                    out_ptr0[static_cast<long>(x1 + (3L*x2) + (147L*x0))] = tmp0;
                }
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
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
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (50176L*x1) + (150528L*x0))];
                        out_ptr1[static_cast<long>(x1 + (3L*x2) + (150528L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_1 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_2 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_3 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_4 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_5 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_6 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_7 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_8 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_9 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_10 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_11 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_12 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_13 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_14 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_15 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_16 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_17 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_18 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_19 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_20 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_21 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_22 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_23 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_24 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_25 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_26 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_27 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_28 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_29 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_30 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_31 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_32 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_33 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_34 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_35 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_36 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_37 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_38 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_39 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_40 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_41 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_42 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_43 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_44 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_45 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_46 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_47 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_48 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_49 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_50 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_51 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_52 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_53 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_54 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_55 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_56 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_57 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_58 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_59 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_60 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_61 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_62 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_63 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_relu_64 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(8192.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_mean_relu_view_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto out_ptr3 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp1);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8192.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x2) + (786432L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp5 = static_cast<float>(8192.0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            auto tmp8 = static_cast<float>(1e-05);
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 + tmp9;
                            auto tmp11 = tmp10.rsqrt();
                            auto tmp12 = tmp3 * tmp11;
                            auto tmp14 = tmp12 * tmp13;
                            auto tmp16 = tmp14 + tmp15;
                            tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        }
                        tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_66 = async_compile.cpp('''
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
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const long* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const long* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       const float* in_ptr10,
                       const long* in_ptr11,
                       const float* in_ptr12,
                       const float* in_ptr13,
                       const float* in_ptr14,
                       const long* in_ptr15,
                       const float* in_ptr16,
                       const float* in_ptr17,
                       const float* in_ptr18,
                       const long* in_ptr19,
                       const float* in_ptr20,
                       const float* in_ptr21,
                       const float* in_ptr22,
                       const long* in_ptr23,
                       const float* in_ptr24,
                       const float* in_ptr25,
                       const float* in_ptr26,
                       const long* in_ptr27,
                       const float* in_ptr28,
                       const float* in_ptr29,
                       const float* in_ptr30,
                       const long* in_ptr31,
                       const float* in_ptr32,
                       const float* in_ptr33,
                       const float* in_ptr34,
                       const long* in_ptr35,
                       const float* in_ptr36,
                       const float* in_ptr37,
                       const float* in_ptr38,
                       const long* in_ptr39,
                       const float* in_ptr40,
                       const float* in_ptr41,
                       const float* in_ptr42,
                       const long* in_ptr43,
                       const float* in_ptr44,
                       const float* in_ptr45,
                       const float* in_ptr46,
                       const long* in_ptr47,
                       const float* in_ptr48,
                       const float* in_ptr49,
                       const float* in_ptr50,
                       const long* in_ptr51,
                       const float* in_ptr52,
                       const float* in_ptr53,
                       const float* in_ptr54,
                       const long* in_ptr55,
                       const float* in_ptr56,
                       const float* in_ptr57,
                       const float* in_ptr58,
                       const long* in_ptr59,
                       const float* in_ptr60,
                       const float* in_ptr61,
                       const float* in_ptr62,
                       const long* in_ptr63,
                       const float* in_ptr64,
                       const float* in_ptr65,
                       const float* in_ptr66,
                       const long* in_ptr67,
                       const float* in_ptr68,
                       const float* in_ptr69,
                       const float* in_ptr70,
                       const long* in_ptr71,
                       const float* in_ptr72,
                       const float* in_ptr73,
                       const float* in_ptr74,
                       const long* in_ptr75,
                       const float* in_ptr76,
                       const float* in_ptr77,
                       const float* in_ptr78,
                       const long* in_ptr79,
                       const float* in_ptr80,
                       const float* in_ptr81,
                       const float* in_ptr82,
                       const long* in_ptr83,
                       const float* in_ptr84,
                       const float* in_ptr85,
                       const float* in_ptr86,
                       const long* in_ptr87,
                       const float* in_ptr88,
                       const float* in_ptr89,
                       const float* in_ptr90,
                       const long* in_ptr91,
                       const float* in_ptr92,
                       const float* in_ptr93,
                       const float* in_ptr94,
                       const long* in_ptr95,
                       const float* in_ptr96,
                       const float* in_ptr97,
                       const float* in_ptr98,
                       const long* in_ptr99,
                       const float* in_ptr100,
                       const float* in_ptr101,
                       const float* in_ptr102,
                       const long* in_ptr103,
                       const float* in_ptr104,
                       const float* in_ptr105,
                       const float* in_ptr106,
                       const long* in_ptr107,
                       const float* in_ptr108,
                       const float* in_ptr109,
                       const float* in_ptr110,
                       const long* in_ptr111,
                       const float* in_ptr112,
                       const float* in_ptr113,
                       const float* in_ptr114,
                       const long* in_ptr115,
                       const float* in_ptr116,
                       const float* in_ptr117,
                       const float* in_ptr118,
                       const long* in_ptr119,
                       const float* in_ptr120,
                       const float* in_ptr121,
                       const float* in_ptr122,
                       const long* in_ptr123,
                       const float* in_ptr124,
                       const float* in_ptr125,
                       const float* in_ptr126,
                       const long* in_ptr127,
                       const float* in_ptr128,
                       const float* in_ptr129,
                       const float* in_ptr130,
                       const long* in_ptr131,
                       const float* in_ptr132,
                       const float* in_ptr133,
                       const float* in_ptr134,
                       const long* in_ptr135,
                       const float* in_ptr136,
                       const float* in_ptr137,
                       const float* in_ptr138,
                       const long* in_ptr139,
                       const float* in_ptr140,
                       const float* in_ptr141,
                       const float* in_ptr142,
                       const long* in_ptr143,
                       const float* in_ptr144,
                       const float* in_ptr145,
                       const float* in_ptr146,
                       const long* in_ptr147,
                       const float* in_ptr148,
                       const float* in_ptr149,
                       const float* in_ptr150,
                       const long* in_ptr151,
                       const float* in_ptr152,
                       const float* in_ptr153,
                       const float* in_ptr154,
                       const long* in_ptr155,
                       const float* in_ptr156,
                       const float* in_ptr157,
                       const float* in_ptr158,
                       const long* in_ptr159,
                       const float* in_ptr160,
                       const float* in_ptr161,
                       const float* in_ptr162,
                       const long* in_ptr163,
                       const float* in_ptr164,
                       const float* in_ptr165,
                       const float* in_ptr166,
                       const long* in_ptr167,
                       const float* in_ptr168,
                       const float* in_ptr169,
                       const float* in_ptr170,
                       const long* in_ptr171,
                       const float* in_ptr172,
                       const float* in_ptr173,
                       const float* in_ptr174,
                       const long* in_ptr175,
                       const float* in_ptr176,
                       const float* in_ptr177,
                       const float* in_ptr178,
                       const long* in_ptr179,
                       const float* in_ptr180,
                       const float* in_ptr181,
                       const float* in_ptr182,
                       const long* in_ptr183,
                       const float* in_ptr184,
                       const float* in_ptr185,
                       const float* in_ptr186,
                       const long* in_ptr187,
                       const float* in_ptr188,
                       const float* in_ptr189,
                       const float* in_ptr190,
                       const long* in_ptr191,
                       const float* in_ptr192,
                       const float* in_ptr193,
                       const float* in_ptr194,
                       const long* in_ptr195,
                       const float* in_ptr196,
                       const float* in_ptr197,
                       const float* in_ptr198,
                       const long* in_ptr199,
                       const float* in_ptr200,
                       const float* in_ptr201,
                       const float* in_ptr202,
                       const long* in_ptr203,
                       const float* in_ptr204,
                       const float* in_ptr205,
                       const float* in_ptr206,
                       const long* in_ptr207,
                       const float* in_ptr208,
                       const float* in_ptr209,
                       const float* in_ptr210,
                       const long* in_ptr211,
                       const float* in_ptr212,
                       const float* in_ptr213,
                       const float* in_ptr214,
                       const long* in_ptr215,
                       const float* in_ptr216,
                       const float* in_ptr217,
                       const float* in_ptr218,
                       const long* in_ptr219,
                       const float* in_ptr220,
                       const float* in_ptr221,
                       const float* in_ptr222,
                       const long* in_ptr223,
                       const float* in_ptr224,
                       const float* in_ptr225,
                       const float* in_ptr226,
                       const long* in_ptr227,
                       const float* in_ptr228,
                       const float* in_ptr229,
                       const float* in_ptr230,
                       const long* in_ptr231,
                       const float* in_ptr232,
                       const float* in_ptr233,
                       const float* in_ptr234,
                       const long* in_ptr235,
                       const float* in_ptr236,
                       const float* in_ptr237,
                       const float* in_ptr238,
                       const long* in_ptr239,
                       const float* in_ptr240,
                       const float* in_ptr241,
                       const float* in_ptr242,
                       const long* in_ptr243,
                       const float* in_ptr244,
                       const float* in_ptr245,
                       const float* in_ptr246,
                       const long* in_ptr247,
                       const float* in_ptr248,
                       const float* in_ptr249,
                       const float* in_ptr250,
                       float* out_ptr1,
                       float* out_ptr2,
                       long* out_ptr4,
                       float* out_ptr6,
                       float* out_ptr7,
                       long* out_ptr9,
                       float* out_ptr11,
                       float* out_ptr12,
                       long* out_ptr14,
                       float* out_ptr16,
                       float* out_ptr17,
                       long* out_ptr19,
                       float* out_ptr21,
                       float* out_ptr22,
                       long* out_ptr24,
                       float* out_ptr26,
                       float* out_ptr27,
                       long* out_ptr29,
                       float* out_ptr31,
                       float* out_ptr32,
                       long* out_ptr34,
                       float* out_ptr36,
                       float* out_ptr37,
                       long* out_ptr39,
                       float* out_ptr41,
                       float* out_ptr42,
                       long* out_ptr44,
                       float* out_ptr46,
                       float* out_ptr47,
                       long* out_ptr49,
                       float* out_ptr51,
                       float* out_ptr52,
                       long* out_ptr54,
                       float* out_ptr56,
                       float* out_ptr57,
                       long* out_ptr59,
                       float* out_ptr61,
                       float* out_ptr62,
                       long* out_ptr64,
                       float* out_ptr66,
                       float* out_ptr67,
                       long* out_ptr69,
                       float* out_ptr71,
                       float* out_ptr72,
                       long* out_ptr74,
                       float* out_ptr76,
                       float* out_ptr77,
                       long* out_ptr79,
                       float* out_ptr81,
                       float* out_ptr82,
                       long* out_ptr84,
                       float* out_ptr86,
                       float* out_ptr87,
                       long* out_ptr89,
                       float* out_ptr91,
                       float* out_ptr92,
                       long* out_ptr94,
                       float* out_ptr96,
                       float* out_ptr97,
                       long* out_ptr99,
                       float* out_ptr101,
                       float* out_ptr102,
                       long* out_ptr104,
                       float* out_ptr106,
                       float* out_ptr107,
                       long* out_ptr109,
                       float* out_ptr111,
                       float* out_ptr112,
                       long* out_ptr114,
                       float* out_ptr116,
                       float* out_ptr117,
                       long* out_ptr119,
                       float* out_ptr121,
                       float* out_ptr122,
                       long* out_ptr124,
                       float* out_ptr126,
                       float* out_ptr127,
                       long* out_ptr129,
                       float* out_ptr131,
                       float* out_ptr132,
                       long* out_ptr134,
                       float* out_ptr136,
                       float* out_ptr137,
                       long* out_ptr139,
                       float* out_ptr141,
                       float* out_ptr142,
                       long* out_ptr144,
                       float* out_ptr146,
                       float* out_ptr147,
                       long* out_ptr149,
                       float* out_ptr151,
                       float* out_ptr152,
                       long* out_ptr154,
                       float* out_ptr156,
                       float* out_ptr157,
                       long* out_ptr159,
                       float* out_ptr161,
                       float* out_ptr162,
                       long* out_ptr164,
                       float* out_ptr166,
                       float* out_ptr167,
                       long* out_ptr169,
                       float* out_ptr171,
                       float* out_ptr172,
                       long* out_ptr174,
                       float* out_ptr176,
                       float* out_ptr177,
                       long* out_ptr179,
                       float* out_ptr181,
                       float* out_ptr182,
                       long* out_ptr184,
                       float* out_ptr186,
                       float* out_ptr187,
                       long* out_ptr189,
                       float* out_ptr191,
                       float* out_ptr192,
                       long* out_ptr194,
                       float* out_ptr196,
                       float* out_ptr197,
                       long* out_ptr199,
                       float* out_ptr201,
                       float* out_ptr202,
                       long* out_ptr204,
                       float* out_ptr206,
                       float* out_ptr207,
                       long* out_ptr209,
                       float* out_ptr211,
                       float* out_ptr212,
                       long* out_ptr214,
                       float* out_ptr216,
                       float* out_ptr217,
                       long* out_ptr219,
                       float* out_ptr221,
                       float* out_ptr222,
                       long* out_ptr224,
                       float* out_ptr226,
                       float* out_ptr227,
                       long* out_ptr229,
                       float* out_ptr231,
                       float* out_ptr232,
                       long* out_ptr234,
                       float* out_ptr236,
                       float* out_ptr237,
                       long* out_ptr239,
                       float* out_ptr241,
                       float* out_ptr242,
                       long* out_ptr244,
                       float* out_ptr246,
                       float* out_ptr247,
                       long* out_ptr249,
                       float* out_ptr251,
                       float* out_ptr252,
                       long* out_ptr254,
                       float* out_ptr256,
                       float* out_ptr257,
                       long* out_ptr259,
                       float* out_ptr261,
                       float* out_ptr262,
                       long* out_ptr264,
                       float* out_ptr266,
                       float* out_ptr267,
                       long* out_ptr269,
                       float* out_ptr271,
                       float* out_ptr272,
                       long* out_ptr274,
                       float* out_ptr276,
                       float* out_ptr277,
                       long* out_ptr279,
                       float* out_ptr281,
                       float* out_ptr282,
                       long* out_ptr284,
                       float* out_ptr286,
                       float* out_ptr287,
                       long* out_ptr289,
                       float* out_ptr291,
                       float* out_ptr292,
                       long* out_ptr294,
                       float* out_ptr296,
                       float* out_ptr297,
                       long* out_ptr299,
                       float* out_ptr301,
                       float* out_ptr302,
                       long* out_ptr304,
                       float* out_ptr306,
                       float* out_ptr307,
                       long* out_ptr309,
                       float* out_ptr311,
                       float* out_ptr312)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr3[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr4[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr6 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr6 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr7 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr7 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr7[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr9[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr11 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr11 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr12 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr12 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr11[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr14[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr16 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr16 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr3 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr17 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr17 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr15[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr19[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr16 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr21 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr21 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr4 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr22 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr22 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr19[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr24[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr20 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr26 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr26 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr5 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr27 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr27 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr23[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr29[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr24 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr31 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr31 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr6 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr32 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr32 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr27[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr34[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr28 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr36 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr36 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr7 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr37 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr37 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr31[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr39[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr32 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr41 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr41 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr8 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr42 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr42 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr35[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr44[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr36 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr46 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr46 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr9 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr47 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr47 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr39[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr49[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr40 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr51 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr51 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr10 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr52 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr52 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr43[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr54[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr44 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr56 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr56 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr11 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr57 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr57 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr47[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr59[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr48 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr61 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr61 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr12 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr62 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr62 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr51[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr64[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr52 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr66 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr66 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr13 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr67 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr67 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr55[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr69[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr56 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr71 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr71 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr14 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr72 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr72 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr59[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr74[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr60 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr76 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr76 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr15 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr77 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr77 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr63[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr79[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr64 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr81 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr81 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr16 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr82 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr82 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr67[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr84[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr68 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr86 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr86 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr17 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr87 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr87 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr71[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr89[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr72 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr91 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr91 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr18 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr92 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr92 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr75[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr94[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr76 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr96 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr96 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr19 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr97 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr97 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr79[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr99[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr80 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr101 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr101 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr20 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr102 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr102 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr83[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr104[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr84 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr106 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr106 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr21 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr107 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr107 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr87[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr109[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr88 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr111 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr111 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr22 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr112 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr112 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr91[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr114[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr92 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr116 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr116 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr23 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr117 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr117 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr95[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr119[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr96 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr121 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr121 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr24 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr122 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr122 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr99[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr124[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr100 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr126 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr126 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr25 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr127 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr127 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr103[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr129[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr104 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr131 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr131 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr26 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr132 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr132 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr107[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr134[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr108 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr136 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr136 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr27 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr137 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr137 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr111[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr139[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr112 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr141 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr141 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr28 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr142 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr142 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr115[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr144[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr116 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr146 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr146 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr29 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr147 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr147 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr119[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr149[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr120 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr151 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr151 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr30 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr152 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr152 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr123[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr154[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr124 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr156 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr156 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr31 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr157 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr157 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr127[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr159[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr128 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr161 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr161 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr32 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr162 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr162 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr131[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr164[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr132 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr166 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr166 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr33 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr167 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr167 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr135[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr169[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr136 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr171 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr171 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr34 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr172 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr172 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr139[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr174[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr140 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr176 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr176 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr35 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr177 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr177 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr143[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr179[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr144 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr181 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr181 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr36 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr182 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr182 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr147[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr184[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr148 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr186 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr186 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr37 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr187 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr187 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr151[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr189[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr152 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr191 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr191 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr38 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr192 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr192 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr155[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr194[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr156 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr196 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr196 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr39 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr197 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr197 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr159[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr199[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr160 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr201 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr201 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr40 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr202 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr202 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr163[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr204[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr164 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr206 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr206 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr41 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr207 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr207 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr167[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr209[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr168 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr211 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr211 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr42 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr212 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr212 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr171[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr214[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr172 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr216 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr216 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr43 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr217 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr217 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr175[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr219[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr176 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr221 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr221 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr44 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr222 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr222 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr179[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr224[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr180 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr226 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr226 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr45 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr227 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr227 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr183[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr229[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr184 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr231 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr231 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr46 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr232 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr232 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr187[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr234[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr188 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr236 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr236 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr47 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr237 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr237 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr191[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr239[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr192 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr241 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr241 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr48 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr242 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr242 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr195[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr244[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr196 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr246 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr246 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr49 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr247 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr247 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr199[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr249[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr200 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr251 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr251 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr50 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr252 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr252 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr203[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr254[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr204 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr256 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr256 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr51 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr257 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr257 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr207[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr259[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr208 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr261 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr261 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr52 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr262 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr262 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr211[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr264[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr212 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr266 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr266 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr53 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr267 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr267 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr215[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr269[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr216 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr271 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr271 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr54 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr272 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr272 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr219[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr274[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr220 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr276 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr276 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr55 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr277 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr277 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr223[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr279[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr224 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr281 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr281 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr56 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr282 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr282 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr227[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr284[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr228 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr286 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr286 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr57 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr287 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr287 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr231[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr289[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr232 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr291 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr291 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr58 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr292 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr292 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr235[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr294[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr236 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr296 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr296 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr59 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr297 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr297 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr239[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr299[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr240 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr301 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr301 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr60 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr302 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr302 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr243[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr304[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr244 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr306 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr306 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr61 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr307 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr307 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr247[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr309[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr248 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr311 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr311 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr62 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr312 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr312 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const long* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const long* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const long* in_ptr8,
                       long* out_ptr1,
                       float* out_ptr3,
                       float* out_ptr4,
                       long* out_ptr6,
                       float* out_ptr8,
                       float* out_ptr9,
                       long* out_ptr11)
{
    {
        auto tmp0 = in_ptr0[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr1[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr3 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr4 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr4[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr6[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr8 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr8 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr9 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8192.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0001220852154804);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr9 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr8[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr11[static_cast<long>(0L)] = tmp2;
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458 = args
    args.clear()
    assert_size_stride(primals_1, (768, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_2, (768, ), (1, ))
    assert_size_stride(primals_3, (768, ), (1, ))
    assert_size_stride(primals_4, (768, ), (1, ))
    assert_size_stride(primals_5, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_6, (768, ), (1, ))
    assert_size_stride(primals_7, (768, ), (1, ))
    assert_size_stride(primals_8, (768, ), (1, ))
    assert_size_stride(primals_9, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_10, (768, ), (1, ))
    assert_size_stride(primals_11, (768, ), (1, ))
    assert_size_stride(primals_12, (768, ), (1, ))
    assert_size_stride(primals_13, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_14, (768, ), (1, ))
    assert_size_stride(primals_15, (768, ), (1, ))
    assert_size_stride(primals_16, (768, ), (1, ))
    assert_size_stride(primals_17, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_18, (768, ), (1, ))
    assert_size_stride(primals_19, (768, ), (1, ))
    assert_size_stride(primals_20, (768, ), (1, ))
    assert_size_stride(primals_21, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_22, (768, ), (1, ))
    assert_size_stride(primals_23, (768, ), (1, ))
    assert_size_stride(primals_24, (768, ), (1, ))
    assert_size_stride(primals_25, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_26, (768, ), (1, ))
    assert_size_stride(primals_27, (768, ), (1, ))
    assert_size_stride(primals_28, (768, ), (1, ))
    assert_size_stride(primals_29, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_30, (768, ), (1, ))
    assert_size_stride(primals_31, (768, ), (1, ))
    assert_size_stride(primals_32, (768, ), (1, ))
    assert_size_stride(primals_33, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_34, (768, ), (1, ))
    assert_size_stride(primals_35, (768, ), (1, ))
    assert_size_stride(primals_36, (768, ), (1, ))
    assert_size_stride(primals_37, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_38, (768, ), (1, ))
    assert_size_stride(primals_39, (768, ), (1, ))
    assert_size_stride(primals_40, (768, ), (1, ))
    assert_size_stride(primals_41, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_42, (768, ), (1, ))
    assert_size_stride(primals_43, (768, ), (1, ))
    assert_size_stride(primals_44, (768, ), (1, ))
    assert_size_stride(primals_45, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_46, (768, ), (1, ))
    assert_size_stride(primals_47, (768, ), (1, ))
    assert_size_stride(primals_48, (768, ), (1, ))
    assert_size_stride(primals_49, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_50, (768, ), (1, ))
    assert_size_stride(primals_51, (768, ), (1, ))
    assert_size_stride(primals_52, (768, ), (1, ))
    assert_size_stride(primals_53, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_54, (768, ), (1, ))
    assert_size_stride(primals_55, (768, ), (1, ))
    assert_size_stride(primals_56, (768, ), (1, ))
    assert_size_stride(primals_57, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_58, (768, ), (1, ))
    assert_size_stride(primals_59, (768, ), (1, ))
    assert_size_stride(primals_60, (768, ), (1, ))
    assert_size_stride(primals_61, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_62, (768, ), (1, ))
    assert_size_stride(primals_63, (768, ), (1, ))
    assert_size_stride(primals_64, (768, ), (1, ))
    assert_size_stride(primals_65, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_66, (768, ), (1, ))
    assert_size_stride(primals_67, (768, ), (1, ))
    assert_size_stride(primals_68, (768, ), (1, ))
    assert_size_stride(primals_69, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_70, (768, ), (1, ))
    assert_size_stride(primals_71, (768, ), (1, ))
    assert_size_stride(primals_72, (768, ), (1, ))
    assert_size_stride(primals_73, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_74, (768, ), (1, ))
    assert_size_stride(primals_75, (768, ), (1, ))
    assert_size_stride(primals_76, (768, ), (1, ))
    assert_size_stride(primals_77, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_78, (768, ), (1, ))
    assert_size_stride(primals_79, (768, ), (1, ))
    assert_size_stride(primals_80, (768, ), (1, ))
    assert_size_stride(primals_81, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_82, (768, ), (1, ))
    assert_size_stride(primals_83, (768, ), (1, ))
    assert_size_stride(primals_84, (768, ), (1, ))
    assert_size_stride(primals_85, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_86, (768, ), (1, ))
    assert_size_stride(primals_87, (768, ), (1, ))
    assert_size_stride(primals_88, (768, ), (1, ))
    assert_size_stride(primals_89, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_90, (768, ), (1, ))
    assert_size_stride(primals_91, (768, ), (1, ))
    assert_size_stride(primals_92, (768, ), (1, ))
    assert_size_stride(primals_93, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_94, (768, ), (1, ))
    assert_size_stride(primals_95, (768, ), (1, ))
    assert_size_stride(primals_96, (768, ), (1, ))
    assert_size_stride(primals_97, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_98, (768, ), (1, ))
    assert_size_stride(primals_99, (768, ), (1, ))
    assert_size_stride(primals_100, (768, ), (1, ))
    assert_size_stride(primals_101, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_102, (768, ), (1, ))
    assert_size_stride(primals_103, (768, ), (1, ))
    assert_size_stride(primals_104, (768, ), (1, ))
    assert_size_stride(primals_105, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_106, (768, ), (1, ))
    assert_size_stride(primals_107, (768, ), (1, ))
    assert_size_stride(primals_108, (768, ), (1, ))
    assert_size_stride(primals_109, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_110, (768, ), (1, ))
    assert_size_stride(primals_111, (768, ), (1, ))
    assert_size_stride(primals_112, (768, ), (1, ))
    assert_size_stride(primals_113, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_114, (768, ), (1, ))
    assert_size_stride(primals_115, (768, ), (1, ))
    assert_size_stride(primals_116, (768, ), (1, ))
    assert_size_stride(primals_117, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_118, (768, ), (1, ))
    assert_size_stride(primals_119, (768, ), (1, ))
    assert_size_stride(primals_120, (768, ), (1, ))
    assert_size_stride(primals_121, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_122, (768, ), (1, ))
    assert_size_stride(primals_123, (768, ), (1, ))
    assert_size_stride(primals_124, (768, ), (1, ))
    assert_size_stride(primals_125, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_126, (768, ), (1, ))
    assert_size_stride(primals_127, (768, ), (1, ))
    assert_size_stride(primals_128, (768, ), (1, ))
    assert_size_stride(primals_129, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_130, (768, ), (1, ))
    assert_size_stride(primals_131, (768, ), (1, ))
    assert_size_stride(primals_132, (768, ), (1, ))
    assert_size_stride(primals_133, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_134, (768, ), (1, ))
    assert_size_stride(primals_135, (768, ), (1, ))
    assert_size_stride(primals_136, (768, ), (1, ))
    assert_size_stride(primals_137, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_138, (768, ), (1, ))
    assert_size_stride(primals_139, (768, ), (1, ))
    assert_size_stride(primals_140, (768, ), (1, ))
    assert_size_stride(primals_141, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_142, (768, ), (1, ))
    assert_size_stride(primals_143, (768, ), (1, ))
    assert_size_stride(primals_144, (768, ), (1, ))
    assert_size_stride(primals_145, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_146, (768, ), (1, ))
    assert_size_stride(primals_147, (768, ), (1, ))
    assert_size_stride(primals_148, (768, ), (1, ))
    assert_size_stride(primals_149, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_150, (768, ), (1, ))
    assert_size_stride(primals_151, (768, ), (1, ))
    assert_size_stride(primals_152, (768, ), (1, ))
    assert_size_stride(primals_153, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_154, (768, ), (1, ))
    assert_size_stride(primals_155, (768, ), (1, ))
    assert_size_stride(primals_156, (768, ), (1, ))
    assert_size_stride(primals_157, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_158, (768, ), (1, ))
    assert_size_stride(primals_159, (768, ), (1, ))
    assert_size_stride(primals_160, (768, ), (1, ))
    assert_size_stride(primals_161, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_162, (768, ), (1, ))
    assert_size_stride(primals_163, (768, ), (1, ))
    assert_size_stride(primals_164, (768, ), (1, ))
    assert_size_stride(primals_165, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_166, (768, ), (1, ))
    assert_size_stride(primals_167, (768, ), (1, ))
    assert_size_stride(primals_168, (768, ), (1, ))
    assert_size_stride(primals_169, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_170, (768, ), (1, ))
    assert_size_stride(primals_171, (768, ), (1, ))
    assert_size_stride(primals_172, (768, ), (1, ))
    assert_size_stride(primals_173, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_174, (768, ), (1, ))
    assert_size_stride(primals_175, (768, ), (1, ))
    assert_size_stride(primals_176, (768, ), (1, ))
    assert_size_stride(primals_177, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_178, (768, ), (1, ))
    assert_size_stride(primals_179, (768, ), (1, ))
    assert_size_stride(primals_180, (768, ), (1, ))
    assert_size_stride(primals_181, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_182, (768, ), (1, ))
    assert_size_stride(primals_183, (768, ), (1, ))
    assert_size_stride(primals_184, (768, ), (1, ))
    assert_size_stride(primals_185, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_186, (768, ), (1, ))
    assert_size_stride(primals_187, (768, ), (1, ))
    assert_size_stride(primals_188, (768, ), (1, ))
    assert_size_stride(primals_189, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_190, (768, ), (1, ))
    assert_size_stride(primals_191, (768, ), (1, ))
    assert_size_stride(primals_192, (768, ), (1, ))
    assert_size_stride(primals_193, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_194, (768, ), (1, ))
    assert_size_stride(primals_195, (768, ), (1, ))
    assert_size_stride(primals_196, (768, ), (1, ))
    assert_size_stride(primals_197, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_198, (768, ), (1, ))
    assert_size_stride(primals_199, (768, ), (1, ))
    assert_size_stride(primals_200, (768, ), (1, ))
    assert_size_stride(primals_201, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_202, (768, ), (1, ))
    assert_size_stride(primals_203, (768, ), (1, ))
    assert_size_stride(primals_204, (768, ), (1, ))
    assert_size_stride(primals_205, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_206, (768, ), (1, ))
    assert_size_stride(primals_207, (768, ), (1, ))
    assert_size_stride(primals_208, (768, ), (1, ))
    assert_size_stride(primals_209, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_210, (768, ), (1, ))
    assert_size_stride(primals_211, (768, ), (1, ))
    assert_size_stride(primals_212, (768, ), (1, ))
    assert_size_stride(primals_213, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_214, (768, ), (1, ))
    assert_size_stride(primals_215, (768, ), (1, ))
    assert_size_stride(primals_216, (768, ), (1, ))
    assert_size_stride(primals_217, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_218, (768, ), (1, ))
    assert_size_stride(primals_219, (768, ), (1, ))
    assert_size_stride(primals_220, (768, ), (1, ))
    assert_size_stride(primals_221, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_222, (768, ), (1, ))
    assert_size_stride(primals_223, (768, ), (1, ))
    assert_size_stride(primals_224, (768, ), (1, ))
    assert_size_stride(primals_225, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_226, (768, ), (1, ))
    assert_size_stride(primals_227, (768, ), (1, ))
    assert_size_stride(primals_228, (768, ), (1, ))
    assert_size_stride(primals_229, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_230, (768, ), (1, ))
    assert_size_stride(primals_231, (768, ), (1, ))
    assert_size_stride(primals_232, (768, ), (1, ))
    assert_size_stride(primals_233, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_234, (768, ), (1, ))
    assert_size_stride(primals_235, (768, ), (1, ))
    assert_size_stride(primals_236, (768, ), (1, ))
    assert_size_stride(primals_237, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_238, (768, ), (1, ))
    assert_size_stride(primals_239, (768, ), (1, ))
    assert_size_stride(primals_240, (768, ), (1, ))
    assert_size_stride(primals_241, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_242, (768, ), (1, ))
    assert_size_stride(primals_243, (768, ), (1, ))
    assert_size_stride(primals_244, (768, ), (1, ))
    assert_size_stride(primals_245, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_246, (768, ), (1, ))
    assert_size_stride(primals_247, (768, ), (1, ))
    assert_size_stride(primals_248, (768, ), (1, ))
    assert_size_stride(primals_249, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_250, (768, ), (1, ))
    assert_size_stride(primals_251, (768, ), (1, ))
    assert_size_stride(primals_252, (768, ), (1, ))
    assert_size_stride(primals_253, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_254, (768, ), (1, ))
    assert_size_stride(primals_255, (768, ), (1, ))
    assert_size_stride(primals_256, (768, ), (1, ))
    assert_size_stride(primals_257, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_258, (768, ), (1, ))
    assert_size_stride(primals_259, (768, ), (1, ))
    assert_size_stride(primals_260, (768, ), (1, ))
    assert_size_stride(primals_261, (1000, 768), (768, 1))
    assert_size_stride(primals_262, (1000, ), (1, ))
    assert_size_stride(primals_263, (768, ), (1, ))
    assert_size_stride(primals_264, (768, ), (1, ))
    assert_size_stride(primals_265, (), ())
    assert_size_stride(primals_266, (768, ), (1, ))
    assert_size_stride(primals_267, (768, ), (1, ))
    assert_size_stride(primals_268, (), ())
    assert_size_stride(primals_269, (768, ), (1, ))
    assert_size_stride(primals_270, (768, ), (1, ))
    assert_size_stride(primals_271, (), ())
    assert_size_stride(primals_272, (768, ), (1, ))
    assert_size_stride(primals_273, (768, ), (1, ))
    assert_size_stride(primals_274, (), ())
    assert_size_stride(primals_275, (768, ), (1, ))
    assert_size_stride(primals_276, (768, ), (1, ))
    assert_size_stride(primals_277, (), ())
    assert_size_stride(primals_278, (768, ), (1, ))
    assert_size_stride(primals_279, (768, ), (1, ))
    assert_size_stride(primals_280, (), ())
    assert_size_stride(primals_281, (768, ), (1, ))
    assert_size_stride(primals_282, (768, ), (1, ))
    assert_size_stride(primals_283, (), ())
    assert_size_stride(primals_284, (768, ), (1, ))
    assert_size_stride(primals_285, (768, ), (1, ))
    assert_size_stride(primals_286, (), ())
    assert_size_stride(primals_287, (768, ), (1, ))
    assert_size_stride(primals_288, (768, ), (1, ))
    assert_size_stride(primals_289, (), ())
    assert_size_stride(primals_290, (768, ), (1, ))
    assert_size_stride(primals_291, (768, ), (1, ))
    assert_size_stride(primals_292, (), ())
    assert_size_stride(primals_293, (768, ), (1, ))
    assert_size_stride(primals_294, (768, ), (1, ))
    assert_size_stride(primals_295, (), ())
    assert_size_stride(primals_296, (768, ), (1, ))
    assert_size_stride(primals_297, (768, ), (1, ))
    assert_size_stride(primals_298, (), ())
    assert_size_stride(primals_299, (768, ), (1, ))
    assert_size_stride(primals_300, (768, ), (1, ))
    assert_size_stride(primals_301, (), ())
    assert_size_stride(primals_302, (768, ), (1, ))
    assert_size_stride(primals_303, (768, ), (1, ))
    assert_size_stride(primals_304, (), ())
    assert_size_stride(primals_305, (768, ), (1, ))
    assert_size_stride(primals_306, (768, ), (1, ))
    assert_size_stride(primals_307, (), ())
    assert_size_stride(primals_308, (768, ), (1, ))
    assert_size_stride(primals_309, (768, ), (1, ))
    assert_size_stride(primals_310, (), ())
    assert_size_stride(primals_311, (768, ), (1, ))
    assert_size_stride(primals_312, (768, ), (1, ))
    assert_size_stride(primals_313, (), ())
    assert_size_stride(primals_314, (768, ), (1, ))
    assert_size_stride(primals_315, (768, ), (1, ))
    assert_size_stride(primals_316, (), ())
    assert_size_stride(primals_317, (768, ), (1, ))
    assert_size_stride(primals_318, (768, ), (1, ))
    assert_size_stride(primals_319, (), ())
    assert_size_stride(primals_320, (768, ), (1, ))
    assert_size_stride(primals_321, (768, ), (1, ))
    assert_size_stride(primals_322, (), ())
    assert_size_stride(primals_323, (768, ), (1, ))
    assert_size_stride(primals_324, (768, ), (1, ))
    assert_size_stride(primals_325, (), ())
    assert_size_stride(primals_326, (768, ), (1, ))
    assert_size_stride(primals_327, (768, ), (1, ))
    assert_size_stride(primals_328, (), ())
    assert_size_stride(primals_329, (768, ), (1, ))
    assert_size_stride(primals_330, (768, ), (1, ))
    assert_size_stride(primals_331, (), ())
    assert_size_stride(primals_332, (768, ), (1, ))
    assert_size_stride(primals_333, (768, ), (1, ))
    assert_size_stride(primals_334, (), ())
    assert_size_stride(primals_335, (768, ), (1, ))
    assert_size_stride(primals_336, (768, ), (1, ))
    assert_size_stride(primals_337, (), ())
    assert_size_stride(primals_338, (768, ), (1, ))
    assert_size_stride(primals_339, (768, ), (1, ))
    assert_size_stride(primals_340, (), ())
    assert_size_stride(primals_341, (768, ), (1, ))
    assert_size_stride(primals_342, (768, ), (1, ))
    assert_size_stride(primals_343, (), ())
    assert_size_stride(primals_344, (768, ), (1, ))
    assert_size_stride(primals_345, (768, ), (1, ))
    assert_size_stride(primals_346, (), ())
    assert_size_stride(primals_347, (768, ), (1, ))
    assert_size_stride(primals_348, (768, ), (1, ))
    assert_size_stride(primals_349, (), ())
    assert_size_stride(primals_350, (768, ), (1, ))
    assert_size_stride(primals_351, (768, ), (1, ))
    assert_size_stride(primals_352, (), ())
    assert_size_stride(primals_353, (768, ), (1, ))
    assert_size_stride(primals_354, (768, ), (1, ))
    assert_size_stride(primals_355, (), ())
    assert_size_stride(primals_356, (768, ), (1, ))
    assert_size_stride(primals_357, (768, ), (1, ))
    assert_size_stride(primals_358, (), ())
    assert_size_stride(primals_359, (768, ), (1, ))
    assert_size_stride(primals_360, (768, ), (1, ))
    assert_size_stride(primals_361, (), ())
    assert_size_stride(primals_362, (768, ), (1, ))
    assert_size_stride(primals_363, (768, ), (1, ))
    assert_size_stride(primals_364, (), ())
    assert_size_stride(primals_365, (768, ), (1, ))
    assert_size_stride(primals_366, (768, ), (1, ))
    assert_size_stride(primals_367, (), ())
    assert_size_stride(primals_368, (768, ), (1, ))
    assert_size_stride(primals_369, (768, ), (1, ))
    assert_size_stride(primals_370, (), ())
    assert_size_stride(primals_371, (768, ), (1, ))
    assert_size_stride(primals_372, (768, ), (1, ))
    assert_size_stride(primals_373, (), ())
    assert_size_stride(primals_374, (768, ), (1, ))
    assert_size_stride(primals_375, (768, ), (1, ))
    assert_size_stride(primals_376, (), ())
    assert_size_stride(primals_377, (768, ), (1, ))
    assert_size_stride(primals_378, (768, ), (1, ))
    assert_size_stride(primals_379, (), ())
    assert_size_stride(primals_380, (768, ), (1, ))
    assert_size_stride(primals_381, (768, ), (1, ))
    assert_size_stride(primals_382, (), ())
    assert_size_stride(primals_383, (768, ), (1, ))
    assert_size_stride(primals_384, (768, ), (1, ))
    assert_size_stride(primals_385, (), ())
    assert_size_stride(primals_386, (768, ), (1, ))
    assert_size_stride(primals_387, (768, ), (1, ))
    assert_size_stride(primals_388, (), ())
    assert_size_stride(primals_389, (768, ), (1, ))
    assert_size_stride(primals_390, (768, ), (1, ))
    assert_size_stride(primals_391, (), ())
    assert_size_stride(primals_392, (768, ), (1, ))
    assert_size_stride(primals_393, (768, ), (1, ))
    assert_size_stride(primals_394, (), ())
    assert_size_stride(primals_395, (768, ), (1, ))
    assert_size_stride(primals_396, (768, ), (1, ))
    assert_size_stride(primals_397, (), ())
    assert_size_stride(primals_398, (768, ), (1, ))
    assert_size_stride(primals_399, (768, ), (1, ))
    assert_size_stride(primals_400, (), ())
    assert_size_stride(primals_401, (768, ), (1, ))
    assert_size_stride(primals_402, (768, ), (1, ))
    assert_size_stride(primals_403, (), ())
    assert_size_stride(primals_404, (768, ), (1, ))
    assert_size_stride(primals_405, (768, ), (1, ))
    assert_size_stride(primals_406, (), ())
    assert_size_stride(primals_407, (768, ), (1, ))
    assert_size_stride(primals_408, (768, ), (1, ))
    assert_size_stride(primals_409, (), ())
    assert_size_stride(primals_410, (768, ), (1, ))
    assert_size_stride(primals_411, (768, ), (1, ))
    assert_size_stride(primals_412, (), ())
    assert_size_stride(primals_413, (768, ), (1, ))
    assert_size_stride(primals_414, (768, ), (1, ))
    assert_size_stride(primals_415, (), ())
    assert_size_stride(primals_416, (768, ), (1, ))
    assert_size_stride(primals_417, (768, ), (1, ))
    assert_size_stride(primals_418, (), ())
    assert_size_stride(primals_419, (768, ), (1, ))
    assert_size_stride(primals_420, (768, ), (1, ))
    assert_size_stride(primals_421, (), ())
    assert_size_stride(primals_422, (768, ), (1, ))
    assert_size_stride(primals_423, (768, ), (1, ))
    assert_size_stride(primals_424, (), ())
    assert_size_stride(primals_425, (768, ), (1, ))
    assert_size_stride(primals_426, (768, ), (1, ))
    assert_size_stride(primals_427, (), ())
    assert_size_stride(primals_428, (768, ), (1, ))
    assert_size_stride(primals_429, (768, ), (1, ))
    assert_size_stride(primals_430, (), ())
    assert_size_stride(primals_431, (768, ), (1, ))
    assert_size_stride(primals_432, (768, ), (1, ))
    assert_size_stride(primals_433, (), ())
    assert_size_stride(primals_434, (768, ), (1, ))
    assert_size_stride(primals_435, (768, ), (1, ))
    assert_size_stride(primals_436, (), ())
    assert_size_stride(primals_437, (768, ), (1, ))
    assert_size_stride(primals_438, (768, ), (1, ))
    assert_size_stride(primals_439, (), ())
    assert_size_stride(primals_440, (768, ), (1, ))
    assert_size_stride(primals_441, (768, ), (1, ))
    assert_size_stride(primals_442, (), ())
    assert_size_stride(primals_443, (768, ), (1, ))
    assert_size_stride(primals_444, (768, ), (1, ))
    assert_size_stride(primals_445, (), ())
    assert_size_stride(primals_446, (768, ), (1, ))
    assert_size_stride(primals_447, (768, ), (1, ))
    assert_size_stride(primals_448, (), ())
    assert_size_stride(primals_449, (768, ), (1, ))
    assert_size_stride(primals_450, (768, ), (1, ))
    assert_size_stride(primals_451, (), ())
    assert_size_stride(primals_452, (768, ), (1, ))
    assert_size_stride(primals_453, (768, ), (1, ))
    assert_size_stride(primals_454, (), ())
    assert_size_stride(primals_455, (768, ), (1, ))
    assert_size_stride(primals_456, (768, ), (1, ))
    assert_size_stride(primals_457, (), ())
    assert_size_stride(primals_458, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((768, 3, 7, 7), (147, 1, 21, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    cpp_fused_0(c_void_p(primals_1.data_ptr()), c_void_p(primals_458.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del primals_1
    del primals_458
    # Source Nodes: [l__mod___stem_0], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf1, buf0, primals_2, stride=(7, 7), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf2, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_2
    buf3 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf6 = empty((768, ), device='cpu', dtype=torch.float32)
    buf7 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_1(c_void_p(buf2.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()))
    del primals_4
    # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___fn_0], Original ATen: [aten.convolution]
    buf8 = extern_kernels.convolution(buf7, primals_5, primals_6, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf8, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_6
    buf9 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf10 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf12 = empty((768, ), device='cpu', dtype=torch.float32)
    buf13 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_2(c_void_p(buf8.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()))
    del primals_8
    # Source Nodes: [l__mod___blocks_0_1], Original ATen: [aten.convolution]
    buf14 = extern_kernels.convolution(buf13, primals_9, primals_10, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf14, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_10
    buf15 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf16 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf18 = empty((768, ), device='cpu', dtype=torch.float32)
    buf19 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_3(c_void_p(buf14.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()))
    del primals_12
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___fn_0], Original ATen: [aten.convolution]
    buf20 = extern_kernels.convolution(buf19, primals_13, primals_14, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf20, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_14
    buf21 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf22 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf24 = empty((768, ), device='cpu', dtype=torch.float32)
    buf25 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_4(c_void_p(buf20.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()))
    del primals_16
    # Source Nodes: [l__mod___blocks_1_1], Original ATen: [aten.convolution]
    buf26 = extern_kernels.convolution(buf25, primals_17, primals_18, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf26, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_18
    buf27 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf28 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf30 = empty((768, ), device='cpu', dtype=torch.float32)
    buf31 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_5(c_void_p(buf26.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()))
    del primals_20
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___fn_0], Original ATen: [aten.convolution]
    buf32 = extern_kernels.convolution(buf31, primals_21, primals_22, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf32, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_22
    buf33 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf34 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf36 = empty((768, ), device='cpu', dtype=torch.float32)
    buf37 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_6(c_void_p(buf32.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()))
    del primals_24
    # Source Nodes: [l__mod___blocks_2_1], Original ATen: [aten.convolution]
    buf38 = extern_kernels.convolution(buf37, primals_25, primals_26, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf38, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_26
    buf39 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf40 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf42 = empty((768, ), device='cpu', dtype=torch.float32)
    buf43 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_7(c_void_p(buf38.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf43.data_ptr()))
    del primals_28
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___fn_0], Original ATen: [aten.convolution]
    buf44 = extern_kernels.convolution(buf43, primals_29, primals_30, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf44, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_30
    buf45 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf46 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf48 = empty((768, ), device='cpu', dtype=torch.float32)
    buf49 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_8(c_void_p(buf44.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf49.data_ptr()))
    del primals_32
    # Source Nodes: [l__mod___blocks_3_1], Original ATen: [aten.convolution]
    buf50 = extern_kernels.convolution(buf49, primals_33, primals_34, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf50, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_34
    buf51 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf52 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf54 = empty((768, ), device='cpu', dtype=torch.float32)
    buf55 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_9(c_void_p(buf50.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()))
    del primals_36
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___fn_0], Original ATen: [aten.convolution]
    buf56 = extern_kernels.convolution(buf55, primals_37, primals_38, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf56, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_38
    buf57 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf58 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf60 = empty((768, ), device='cpu', dtype=torch.float32)
    buf61 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_10(c_void_p(buf56.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(primals_40.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()))
    del primals_40
    # Source Nodes: [l__mod___blocks_4_1], Original ATen: [aten.convolution]
    buf62 = extern_kernels.convolution(buf61, primals_41, primals_42, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf62, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_42
    buf63 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf64 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf66 = empty((768, ), device='cpu', dtype=torch.float32)
    buf67 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_11(c_void_p(buf62.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(primals_44.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()))
    del primals_44
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___fn_0], Original ATen: [aten.convolution]
    buf68 = extern_kernels.convolution(buf67, primals_45, primals_46, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf68, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_46
    buf69 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf70 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf72 = empty((768, ), device='cpu', dtype=torch.float32)
    buf73 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_12(c_void_p(buf68.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(primals_48.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf73.data_ptr()))
    del primals_48
    # Source Nodes: [l__mod___blocks_5_1], Original ATen: [aten.convolution]
    buf74 = extern_kernels.convolution(buf73, primals_49, primals_50, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf74, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_50
    buf75 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf76 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf78 = empty((768, ), device='cpu', dtype=torch.float32)
    buf79 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_13(c_void_p(buf74.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(primals_52.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf79.data_ptr()))
    del primals_52
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___fn_0], Original ATen: [aten.convolution]
    buf80 = extern_kernels.convolution(buf79, primals_53, primals_54, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf80, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_54
    buf81 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf82 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf84 = empty((768, ), device='cpu', dtype=torch.float32)
    buf85 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_14(c_void_p(buf80.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(primals_56.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf85.data_ptr()))
    del primals_56
    # Source Nodes: [l__mod___blocks_6_1], Original ATen: [aten.convolution]
    buf86 = extern_kernels.convolution(buf85, primals_57, primals_58, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf86, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_58
    buf87 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf88 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf90 = empty((768, ), device='cpu', dtype=torch.float32)
    buf91 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_15(c_void_p(buf86.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(primals_60.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf91.data_ptr()))
    del primals_60
    # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___fn_0], Original ATen: [aten.convolution]
    buf92 = extern_kernels.convolution(buf91, primals_61, primals_62, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf92, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_62
    buf93 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf94 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf96 = empty((768, ), device='cpu', dtype=torch.float32)
    buf97 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_16(c_void_p(buf92.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(primals_64.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf97.data_ptr()))
    del primals_64
    # Source Nodes: [l__mod___blocks_7_1], Original ATen: [aten.convolution]
    buf98 = extern_kernels.convolution(buf97, primals_65, primals_66, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf98, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_66
    buf99 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf100 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf102 = empty((768, ), device='cpu', dtype=torch.float32)
    buf103 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_17(c_void_p(buf98.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()))
    del primals_68
    # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___fn_0], Original ATen: [aten.convolution]
    buf104 = extern_kernels.convolution(buf103, primals_69, primals_70, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf104, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_70
    buf105 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf106 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf108 = empty((768, ), device='cpu', dtype=torch.float32)
    buf109 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_18(c_void_p(buf104.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(primals_72.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()))
    del primals_72
    # Source Nodes: [l__mod___blocks_8_1], Original ATen: [aten.convolution]
    buf110 = extern_kernels.convolution(buf109, primals_73, primals_74, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf110, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_74
    buf111 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf112 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf114 = empty((768, ), device='cpu', dtype=torch.float32)
    buf115 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_19(c_void_p(buf110.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(primals_76.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf115.data_ptr()))
    del primals_76
    # Source Nodes: [getattr_getattr_l__mod___blocks___9_____0___fn_0], Original ATen: [aten.convolution]
    buf116 = extern_kernels.convolution(buf115, primals_77, primals_78, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf116, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_78
    buf117 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf118 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf120 = empty((768, ), device='cpu', dtype=torch.float32)
    buf121 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_20(c_void_p(buf116.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf121.data_ptr()))
    del primals_80
    # Source Nodes: [l__mod___blocks_9_1], Original ATen: [aten.convolution]
    buf122 = extern_kernels.convolution(buf121, primals_81, primals_82, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf122, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_82
    buf123 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf124 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf126 = empty((768, ), device='cpu', dtype=torch.float32)
    buf127 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_21(c_void_p(buf122.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(primals_84.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf127.data_ptr()))
    del primals_84
    # Source Nodes: [getattr_getattr_l__mod___blocks___10_____0___fn_0], Original ATen: [aten.convolution]
    buf128 = extern_kernels.convolution(buf127, primals_85, primals_86, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf128, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_86
    buf129 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf130 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf132 = empty((768, ), device='cpu', dtype=torch.float32)
    buf133 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_22(c_void_p(buf128.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(primals_88.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf133.data_ptr()))
    del primals_88
    # Source Nodes: [l__mod___blocks_10_1], Original ATen: [aten.convolution]
    buf134 = extern_kernels.convolution(buf133, primals_89, primals_90, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf134, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_90
    buf135 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf136 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf138 = empty((768, ), device='cpu', dtype=torch.float32)
    buf139 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_23(c_void_p(buf134.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(primals_92.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf139.data_ptr()))
    del primals_92
    # Source Nodes: [getattr_getattr_l__mod___blocks___11_____0___fn_0], Original ATen: [aten.convolution]
    buf140 = extern_kernels.convolution(buf139, primals_93, primals_94, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf140, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_94
    buf141 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf142 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf144 = empty((768, ), device='cpu', dtype=torch.float32)
    buf145 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_24(c_void_p(buf140.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(primals_96.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()))
    del primals_96
    # Source Nodes: [l__mod___blocks_11_1], Original ATen: [aten.convolution]
    buf146 = extern_kernels.convolution(buf145, primals_97, primals_98, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf146, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_98
    buf147 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf148 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf150 = empty((768, ), device='cpu', dtype=torch.float32)
    buf151 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_25(c_void_p(buf146.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(primals_100.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf151.data_ptr()))
    del primals_100
    # Source Nodes: [getattr_getattr_l__mod___blocks___12_____0___fn_0], Original ATen: [aten.convolution]
    buf152 = extern_kernels.convolution(buf151, primals_101, primals_102, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf152, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_102
    buf153 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf154 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf156 = empty((768, ), device='cpu', dtype=torch.float32)
    buf157 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_26(c_void_p(buf152.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf157.data_ptr()))
    del primals_104
    # Source Nodes: [l__mod___blocks_12_1], Original ATen: [aten.convolution]
    buf158 = extern_kernels.convolution(buf157, primals_105, primals_106, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf158, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_106
    buf159 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf160 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf162 = empty((768, ), device='cpu', dtype=torch.float32)
    buf163 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_27(c_void_p(buf158.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(primals_108.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()))
    del primals_108
    # Source Nodes: [getattr_getattr_l__mod___blocks___13_____0___fn_0], Original ATen: [aten.convolution]
    buf164 = extern_kernels.convolution(buf163, primals_109, primals_110, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf164, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_110
    buf165 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf166 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf168 = empty((768, ), device='cpu', dtype=torch.float32)
    buf169 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_28(c_void_p(buf164.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(primals_112.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()))
    del primals_112
    # Source Nodes: [l__mod___blocks_13_1], Original ATen: [aten.convolution]
    buf170 = extern_kernels.convolution(buf169, primals_113, primals_114, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf170, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_114
    buf171 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf172 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf174 = empty((768, ), device='cpu', dtype=torch.float32)
    buf175 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_29(c_void_p(buf170.data_ptr()), c_void_p(primals_115.data_ptr()), c_void_p(primals_116.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf175.data_ptr()))
    del primals_116
    # Source Nodes: [getattr_getattr_l__mod___blocks___14_____0___fn_0], Original ATen: [aten.convolution]
    buf176 = extern_kernels.convolution(buf175, primals_117, primals_118, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf176, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_118
    buf177 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf178 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf180 = empty((768, ), device='cpu', dtype=torch.float32)
    buf181 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_30(c_void_p(buf176.data_ptr()), c_void_p(primals_119.data_ptr()), c_void_p(primals_120.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf181.data_ptr()))
    del primals_120
    # Source Nodes: [l__mod___blocks_14_1], Original ATen: [aten.convolution]
    buf182 = extern_kernels.convolution(buf181, primals_121, primals_122, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf182, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_122
    buf183 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf184 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf186 = empty((768, ), device='cpu', dtype=torch.float32)
    buf187 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_31(c_void_p(buf182.data_ptr()), c_void_p(primals_123.data_ptr()), c_void_p(primals_124.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf187.data_ptr()))
    del primals_124
    # Source Nodes: [getattr_getattr_l__mod___blocks___15_____0___fn_0], Original ATen: [aten.convolution]
    buf188 = extern_kernels.convolution(buf187, primals_125, primals_126, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf188, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_126
    buf189 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf190 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf192 = empty((768, ), device='cpu', dtype=torch.float32)
    buf193 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_32(c_void_p(buf188.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(primals_128.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf193.data_ptr()))
    del primals_128
    # Source Nodes: [l__mod___blocks_15_1], Original ATen: [aten.convolution]
    buf194 = extern_kernels.convolution(buf193, primals_129, primals_130, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf194, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_130
    buf195 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf196 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf198 = empty((768, ), device='cpu', dtype=torch.float32)
    buf199 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_33(c_void_p(buf194.data_ptr()), c_void_p(primals_131.data_ptr()), c_void_p(primals_132.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf199.data_ptr()))
    del primals_132
    # Source Nodes: [getattr_getattr_l__mod___blocks___16_____0___fn_0], Original ATen: [aten.convolution]
    buf200 = extern_kernels.convolution(buf199, primals_133, primals_134, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf200, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_134
    buf201 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf202 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf204 = empty((768, ), device='cpu', dtype=torch.float32)
    buf205 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_34(c_void_p(buf200.data_ptr()), c_void_p(primals_135.data_ptr()), c_void_p(primals_136.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf205.data_ptr()))
    del primals_136
    # Source Nodes: [l__mod___blocks_16_1], Original ATen: [aten.convolution]
    buf206 = extern_kernels.convolution(buf205, primals_137, primals_138, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf206, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_138
    buf207 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf208 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf210 = empty((768, ), device='cpu', dtype=torch.float32)
    buf211 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_35(c_void_p(buf206.data_ptr()), c_void_p(primals_139.data_ptr()), c_void_p(primals_140.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf211.data_ptr()))
    del primals_140
    # Source Nodes: [getattr_getattr_l__mod___blocks___17_____0___fn_0], Original ATen: [aten.convolution]
    buf212 = extern_kernels.convolution(buf211, primals_141, primals_142, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf212, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_142
    buf213 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf214 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf216 = empty((768, ), device='cpu', dtype=torch.float32)
    buf217 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_36(c_void_p(buf212.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(primals_144.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf217.data_ptr()))
    del primals_144
    # Source Nodes: [l__mod___blocks_17_1], Original ATen: [aten.convolution]
    buf218 = extern_kernels.convolution(buf217, primals_145, primals_146, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf218, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_146
    buf219 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf220 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf222 = empty((768, ), device='cpu', dtype=torch.float32)
    buf223 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_37(c_void_p(buf218.data_ptr()), c_void_p(primals_147.data_ptr()), c_void_p(primals_148.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf223.data_ptr()))
    del primals_148
    # Source Nodes: [getattr_getattr_l__mod___blocks___18_____0___fn_0], Original ATen: [aten.convolution]
    buf224 = extern_kernels.convolution(buf223, primals_149, primals_150, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf224, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_150
    buf225 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf226 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf228 = empty((768, ), device='cpu', dtype=torch.float32)
    buf229 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_38(c_void_p(buf224.data_ptr()), c_void_p(primals_151.data_ptr()), c_void_p(primals_152.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf229.data_ptr()))
    del primals_152
    # Source Nodes: [l__mod___blocks_18_1], Original ATen: [aten.convolution]
    buf230 = extern_kernels.convolution(buf229, primals_153, primals_154, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf230, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_154
    buf231 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf232 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf234 = empty((768, ), device='cpu', dtype=torch.float32)
    buf235 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_39(c_void_p(buf230.data_ptr()), c_void_p(primals_155.data_ptr()), c_void_p(primals_156.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf235.data_ptr()))
    del primals_156
    # Source Nodes: [getattr_getattr_l__mod___blocks___19_____0___fn_0], Original ATen: [aten.convolution]
    buf236 = extern_kernels.convolution(buf235, primals_157, primals_158, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf236, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_158
    buf237 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf238 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf240 = empty((768, ), device='cpu', dtype=torch.float32)
    buf241 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_40(c_void_p(buf236.data_ptr()), c_void_p(primals_159.data_ptr()), c_void_p(primals_160.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf241.data_ptr()))
    del primals_160
    # Source Nodes: [l__mod___blocks_19_1], Original ATen: [aten.convolution]
    buf242 = extern_kernels.convolution(buf241, primals_161, primals_162, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf242, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_162
    buf243 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf244 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf246 = empty((768, ), device='cpu', dtype=torch.float32)
    buf247 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_41(c_void_p(buf242.data_ptr()), c_void_p(primals_163.data_ptr()), c_void_p(primals_164.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf247.data_ptr()))
    del primals_164
    # Source Nodes: [getattr_getattr_l__mod___blocks___20_____0___fn_0], Original ATen: [aten.convolution]
    buf248 = extern_kernels.convolution(buf247, primals_165, primals_166, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf248, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_166
    buf249 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf250 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf252 = empty((768, ), device='cpu', dtype=torch.float32)
    buf253 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_42(c_void_p(buf248.data_ptr()), c_void_p(primals_167.data_ptr()), c_void_p(primals_168.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf253.data_ptr()))
    del primals_168
    # Source Nodes: [l__mod___blocks_20_1], Original ATen: [aten.convolution]
    buf254 = extern_kernels.convolution(buf253, primals_169, primals_170, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf254, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_170
    buf255 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf256 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf258 = empty((768, ), device='cpu', dtype=torch.float32)
    buf259 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_43(c_void_p(buf254.data_ptr()), c_void_p(primals_171.data_ptr()), c_void_p(primals_172.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf259.data_ptr()))
    del primals_172
    # Source Nodes: [getattr_getattr_l__mod___blocks___21_____0___fn_0], Original ATen: [aten.convolution]
    buf260 = extern_kernels.convolution(buf259, primals_173, primals_174, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf260, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_174
    buf261 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf262 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf264 = empty((768, ), device='cpu', dtype=torch.float32)
    buf265 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_44(c_void_p(buf260.data_ptr()), c_void_p(primals_175.data_ptr()), c_void_p(primals_176.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf265.data_ptr()))
    del primals_176
    # Source Nodes: [l__mod___blocks_21_1], Original ATen: [aten.convolution]
    buf266 = extern_kernels.convolution(buf265, primals_177, primals_178, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf266, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_178
    buf267 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf268 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf270 = empty((768, ), device='cpu', dtype=torch.float32)
    buf271 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_45(c_void_p(buf266.data_ptr()), c_void_p(primals_179.data_ptr()), c_void_p(primals_180.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf271.data_ptr()))
    del primals_180
    # Source Nodes: [getattr_getattr_l__mod___blocks___22_____0___fn_0], Original ATen: [aten.convolution]
    buf272 = extern_kernels.convolution(buf271, primals_181, primals_182, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf272, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_182
    buf273 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf274 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf276 = empty((768, ), device='cpu', dtype=torch.float32)
    buf277 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_46(c_void_p(buf272.data_ptr()), c_void_p(primals_183.data_ptr()), c_void_p(primals_184.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf277.data_ptr()))
    del primals_184
    # Source Nodes: [l__mod___blocks_22_1], Original ATen: [aten.convolution]
    buf278 = extern_kernels.convolution(buf277, primals_185, primals_186, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf278, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_186
    buf279 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf280 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf282 = empty((768, ), device='cpu', dtype=torch.float32)
    buf283 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_47(c_void_p(buf278.data_ptr()), c_void_p(primals_187.data_ptr()), c_void_p(primals_188.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf283.data_ptr()))
    del primals_188
    # Source Nodes: [getattr_getattr_l__mod___blocks___23_____0___fn_0], Original ATen: [aten.convolution]
    buf284 = extern_kernels.convolution(buf283, primals_189, primals_190, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf284, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_190
    buf285 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf286 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf288 = empty((768, ), device='cpu', dtype=torch.float32)
    buf289 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_48(c_void_p(buf284.data_ptr()), c_void_p(primals_191.data_ptr()), c_void_p(primals_192.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf289.data_ptr()))
    del primals_192
    # Source Nodes: [l__mod___blocks_23_1], Original ATen: [aten.convolution]
    buf290 = extern_kernels.convolution(buf289, primals_193, primals_194, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf290, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_194
    buf291 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf292 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf294 = empty((768, ), device='cpu', dtype=torch.float32)
    buf295 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_49(c_void_p(buf290.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(primals_196.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf295.data_ptr()))
    del primals_196
    # Source Nodes: [getattr_getattr_l__mod___blocks___24_____0___fn_0], Original ATen: [aten.convolution]
    buf296 = extern_kernels.convolution(buf295, primals_197, primals_198, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf296, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_198
    buf297 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf298 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf300 = empty((768, ), device='cpu', dtype=torch.float32)
    buf301 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_50(c_void_p(buf296.data_ptr()), c_void_p(primals_199.data_ptr()), c_void_p(primals_200.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf301.data_ptr()))
    del primals_200
    # Source Nodes: [l__mod___blocks_24_1], Original ATen: [aten.convolution]
    buf302 = extern_kernels.convolution(buf301, primals_201, primals_202, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf302, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_202
    buf303 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf304 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf306 = empty((768, ), device='cpu', dtype=torch.float32)
    buf307 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_51(c_void_p(buf302.data_ptr()), c_void_p(primals_203.data_ptr()), c_void_p(primals_204.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf307.data_ptr()))
    del primals_204
    # Source Nodes: [getattr_getattr_l__mod___blocks___25_____0___fn_0], Original ATen: [aten.convolution]
    buf308 = extern_kernels.convolution(buf307, primals_205, primals_206, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf308, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_206
    buf309 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf310 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf312 = empty((768, ), device='cpu', dtype=torch.float32)
    buf313 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_52(c_void_p(buf308.data_ptr()), c_void_p(primals_207.data_ptr()), c_void_p(primals_208.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf313.data_ptr()))
    del primals_208
    # Source Nodes: [l__mod___blocks_25_1], Original ATen: [aten.convolution]
    buf314 = extern_kernels.convolution(buf313, primals_209, primals_210, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf314, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_210
    buf315 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf316 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf318 = empty((768, ), device='cpu', dtype=torch.float32)
    buf319 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_53(c_void_p(buf314.data_ptr()), c_void_p(primals_211.data_ptr()), c_void_p(primals_212.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(buf319.data_ptr()))
    del primals_212
    # Source Nodes: [getattr_getattr_l__mod___blocks___26_____0___fn_0], Original ATen: [aten.convolution]
    buf320 = extern_kernels.convolution(buf319, primals_213, primals_214, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf320, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_214
    buf321 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf322 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf324 = empty((768, ), device='cpu', dtype=torch.float32)
    buf325 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_54(c_void_p(buf320.data_ptr()), c_void_p(primals_215.data_ptr()), c_void_p(primals_216.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf325.data_ptr()))
    del primals_216
    # Source Nodes: [l__mod___blocks_26_1], Original ATen: [aten.convolution]
    buf326 = extern_kernels.convolution(buf325, primals_217, primals_218, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf326, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_218
    buf327 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf328 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf330 = empty((768, ), device='cpu', dtype=torch.float32)
    buf331 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_55(c_void_p(buf326.data_ptr()), c_void_p(primals_219.data_ptr()), c_void_p(primals_220.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf331.data_ptr()))
    del primals_220
    # Source Nodes: [getattr_getattr_l__mod___blocks___27_____0___fn_0], Original ATen: [aten.convolution]
    buf332 = extern_kernels.convolution(buf331, primals_221, primals_222, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf332, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_222
    buf333 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf334 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf336 = empty((768, ), device='cpu', dtype=torch.float32)
    buf337 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_56(c_void_p(buf332.data_ptr()), c_void_p(primals_223.data_ptr()), c_void_p(primals_224.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(buf337.data_ptr()))
    del primals_224
    # Source Nodes: [l__mod___blocks_27_1], Original ATen: [aten.convolution]
    buf338 = extern_kernels.convolution(buf337, primals_225, primals_226, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf338, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_226
    buf339 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf340 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf342 = empty((768, ), device='cpu', dtype=torch.float32)
    buf343 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_57(c_void_p(buf338.data_ptr()), c_void_p(primals_227.data_ptr()), c_void_p(primals_228.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf343.data_ptr()))
    del primals_228
    # Source Nodes: [getattr_getattr_l__mod___blocks___28_____0___fn_0], Original ATen: [aten.convolution]
    buf344 = extern_kernels.convolution(buf343, primals_229, primals_230, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf344, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_230
    buf345 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf346 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf348 = empty((768, ), device='cpu', dtype=torch.float32)
    buf349 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_58(c_void_p(buf344.data_ptr()), c_void_p(primals_231.data_ptr()), c_void_p(primals_232.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf349.data_ptr()))
    del primals_232
    # Source Nodes: [l__mod___blocks_28_1], Original ATen: [aten.convolution]
    buf350 = extern_kernels.convolution(buf349, primals_233, primals_234, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf350, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_234
    buf351 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf352 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf354 = empty((768, ), device='cpu', dtype=torch.float32)
    buf355 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_59(c_void_p(buf350.data_ptr()), c_void_p(primals_235.data_ptr()), c_void_p(primals_236.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf355.data_ptr()))
    del primals_236
    # Source Nodes: [getattr_getattr_l__mod___blocks___29_____0___fn_0], Original ATen: [aten.convolution]
    buf356 = extern_kernels.convolution(buf355, primals_237, primals_238, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf356, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_238
    buf357 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf358 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf360 = empty((768, ), device='cpu', dtype=torch.float32)
    buf361 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_60(c_void_p(buf356.data_ptr()), c_void_p(primals_239.data_ptr()), c_void_p(primals_240.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf361.data_ptr()))
    del primals_240
    # Source Nodes: [l__mod___blocks_29_1], Original ATen: [aten.convolution]
    buf362 = extern_kernels.convolution(buf361, primals_241, primals_242, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf362, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_242
    buf363 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf364 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf366 = empty((768, ), device='cpu', dtype=torch.float32)
    buf367 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_61(c_void_p(buf362.data_ptr()), c_void_p(primals_243.data_ptr()), c_void_p(primals_244.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf367.data_ptr()))
    del primals_244
    # Source Nodes: [getattr_getattr_l__mod___blocks___30_____0___fn_0], Original ATen: [aten.convolution]
    buf368 = extern_kernels.convolution(buf367, primals_245, primals_246, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf368, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_246
    buf369 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf370 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf372 = empty((768, ), device='cpu', dtype=torch.float32)
    buf373 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_62(c_void_p(buf368.data_ptr()), c_void_p(primals_247.data_ptr()), c_void_p(primals_248.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(buf373.data_ptr()))
    del primals_248
    # Source Nodes: [l__mod___blocks_30_1], Original ATen: [aten.convolution]
    buf374 = extern_kernels.convolution(buf373, primals_249, primals_250, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf374, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_250
    buf375 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf376 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf378 = empty((768, ), device='cpu', dtype=torch.float32)
    buf379 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_63(c_void_p(buf374.data_ptr()), c_void_p(primals_251.data_ptr()), c_void_p(primals_252.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf379.data_ptr()))
    del primals_252
    # Source Nodes: [getattr_getattr_l__mod___blocks___31_____0___fn_0], Original ATen: [aten.convolution]
    buf380 = extern_kernels.convolution(buf379, primals_253, primals_254, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768)
    assert_size_stride(buf380, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_254
    buf381 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf382 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf384 = empty((768, ), device='cpu', dtype=torch.float32)
    buf385 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_relu_64(c_void_p(buf380.data_ptr()), c_void_p(primals_255.data_ptr()), c_void_p(primals_256.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf385.data_ptr()))
    del primals_256
    # Source Nodes: [l__mod___blocks_31_1], Original ATen: [aten.convolution]
    buf386 = extern_kernels.convolution(buf385, primals_257, primals_258, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf386, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del primals_258
    buf387 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf388 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf390 = empty((768, ), device='cpu', dtype=torch.float32)
    buf391 = empty_strided((8, 768, 1, 1), (768, 1, 6144, 6144), device='cpu', dtype=torch.float32)
    buf392 = reinterpret_tensor(buf391, (8, 768), (768, 1), 0); del buf391  # reuse
    cpp_fused__native_batch_norm_legit_functional_mean_relu_view_65(c_void_p(buf392.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(primals_259.data_ptr()), c_void_p(primals_260.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf390.data_ptr()))
    del primals_260
    buf393 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [pred], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_262, buf392, reinterpret_tensor(primals_261, (768, 1000), (1, 768), 0), alpha=1, beta=1, out=buf393)
    del primals_262
    buf398 = reinterpret_tensor(buf4, (768, ), (1, ), 0); del buf4  # reuse
    buf406 = reinterpret_tensor(buf10, (768, ), (1, ), 0); del buf10  # reuse
    buf414 = reinterpret_tensor(buf16, (768, ), (1, ), 0); del buf16  # reuse
    buf422 = reinterpret_tensor(buf22, (768, ), (1, ), 0); del buf22  # reuse
    buf430 = reinterpret_tensor(buf28, (768, ), (1, ), 0); del buf28  # reuse
    buf438 = reinterpret_tensor(buf34, (768, ), (1, ), 0); del buf34  # reuse
    buf446 = reinterpret_tensor(buf40, (768, ), (1, ), 0); del buf40  # reuse
    buf454 = reinterpret_tensor(buf46, (768, ), (1, ), 0); del buf46  # reuse
    buf462 = reinterpret_tensor(buf52, (768, ), (1, ), 0); del buf52  # reuse
    buf470 = reinterpret_tensor(buf58, (768, ), (1, ), 0); del buf58  # reuse
    buf478 = reinterpret_tensor(buf64, (768, ), (1, ), 0); del buf64  # reuse
    buf486 = reinterpret_tensor(buf70, (768, ), (1, ), 0); del buf70  # reuse
    buf494 = reinterpret_tensor(buf76, (768, ), (1, ), 0); del buf76  # reuse
    buf502 = reinterpret_tensor(buf82, (768, ), (1, ), 0); del buf82  # reuse
    buf510 = reinterpret_tensor(buf88, (768, ), (1, ), 0); del buf88  # reuse
    buf518 = reinterpret_tensor(buf94, (768, ), (1, ), 0); del buf94  # reuse
    buf526 = reinterpret_tensor(buf100, (768, ), (1, ), 0); del buf100  # reuse
    buf534 = reinterpret_tensor(buf106, (768, ), (1, ), 0); del buf106  # reuse
    buf542 = reinterpret_tensor(buf112, (768, ), (1, ), 0); del buf112  # reuse
    buf550 = reinterpret_tensor(buf118, (768, ), (1, ), 0); del buf118  # reuse
    buf558 = reinterpret_tensor(buf124, (768, ), (1, ), 0); del buf124  # reuse
    buf566 = reinterpret_tensor(buf130, (768, ), (1, ), 0); del buf130  # reuse
    buf574 = reinterpret_tensor(buf136, (768, ), (1, ), 0); del buf136  # reuse
    buf582 = reinterpret_tensor(buf142, (768, ), (1, ), 0); del buf142  # reuse
    buf590 = reinterpret_tensor(buf148, (768, ), (1, ), 0); del buf148  # reuse
    buf598 = reinterpret_tensor(buf154, (768, ), (1, ), 0); del buf154  # reuse
    buf606 = reinterpret_tensor(buf160, (768, ), (1, ), 0); del buf160  # reuse
    buf614 = reinterpret_tensor(buf166, (768, ), (1, ), 0); del buf166  # reuse
    buf622 = reinterpret_tensor(buf172, (768, ), (1, ), 0); del buf172  # reuse
    buf630 = reinterpret_tensor(buf178, (768, ), (1, ), 0); del buf178  # reuse
    buf638 = reinterpret_tensor(buf184, (768, ), (1, ), 0); del buf184  # reuse
    buf646 = reinterpret_tensor(buf190, (768, ), (1, ), 0); del buf190  # reuse
    buf654 = reinterpret_tensor(buf196, (768, ), (1, ), 0); del buf196  # reuse
    buf662 = reinterpret_tensor(buf202, (768, ), (1, ), 0); del buf202  # reuse
    buf670 = reinterpret_tensor(buf208, (768, ), (1, ), 0); del buf208  # reuse
    buf678 = reinterpret_tensor(buf214, (768, ), (1, ), 0); del buf214  # reuse
    buf686 = reinterpret_tensor(buf220, (768, ), (1, ), 0); del buf220  # reuse
    buf694 = reinterpret_tensor(buf226, (768, ), (1, ), 0); del buf226  # reuse
    buf702 = reinterpret_tensor(buf232, (768, ), (1, ), 0); del buf232  # reuse
    buf710 = reinterpret_tensor(buf238, (768, ), (1, ), 0); del buf238  # reuse
    buf718 = reinterpret_tensor(buf244, (768, ), (1, ), 0); del buf244  # reuse
    buf726 = reinterpret_tensor(buf250, (768, ), (1, ), 0); del buf250  # reuse
    buf734 = reinterpret_tensor(buf256, (768, ), (1, ), 0); del buf256  # reuse
    buf742 = reinterpret_tensor(buf262, (768, ), (1, ), 0); del buf262  # reuse
    buf750 = reinterpret_tensor(buf268, (768, ), (1, ), 0); del buf268  # reuse
    buf758 = reinterpret_tensor(buf274, (768, ), (1, ), 0); del buf274  # reuse
    buf766 = reinterpret_tensor(buf280, (768, ), (1, ), 0); del buf280  # reuse
    buf774 = reinterpret_tensor(buf286, (768, ), (1, ), 0); del buf286  # reuse
    buf782 = reinterpret_tensor(buf292, (768, ), (1, ), 0); del buf292  # reuse
    buf790 = reinterpret_tensor(buf298, (768, ), (1, ), 0); del buf298  # reuse
    buf798 = reinterpret_tensor(buf304, (768, ), (1, ), 0); del buf304  # reuse
    buf806 = reinterpret_tensor(buf310, (768, ), (1, ), 0); del buf310  # reuse
    buf814 = reinterpret_tensor(buf316, (768, ), (1, ), 0); del buf316  # reuse
    buf822 = reinterpret_tensor(buf322, (768, ), (1, ), 0); del buf322  # reuse
    buf830 = reinterpret_tensor(buf328, (768, ), (1, ), 0); del buf328  # reuse
    buf838 = reinterpret_tensor(buf334, (768, ), (1, ), 0); del buf334  # reuse
    buf846 = reinterpret_tensor(buf340, (768, ), (1, ), 0); del buf340  # reuse
    buf854 = reinterpret_tensor(buf346, (768, ), (1, ), 0); del buf346  # reuse
    buf862 = reinterpret_tensor(buf352, (768, ), (1, ), 0); del buf352  # reuse
    buf870 = reinterpret_tensor(buf358, (768, ), (1, ), 0); del buf358  # reuse
    buf878 = reinterpret_tensor(buf364, (768, ), (1, ), 0); del buf364  # reuse
    buf886 = reinterpret_tensor(buf370, (768, ), (1, ), 0); del buf370  # reuse
    buf894 = reinterpret_tensor(buf376, (768, ), (1, ), 0); del buf376  # reuse
    cpp_fused__native_batch_norm_legit_functional_add_66(c_void_p(buf398.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(buf446.data_ptr()), c_void_p(buf454.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf478.data_ptr()), c_void_p(buf486.data_ptr()), c_void_p(buf494.data_ptr()), c_void_p(buf502.data_ptr()), c_void_p(buf510.data_ptr()), c_void_p(buf518.data_ptr()), c_void_p(buf526.data_ptr()), c_void_p(buf534.data_ptr()), c_void_p(buf542.data_ptr()), c_void_p(buf550.data_ptr()), c_void_p(buf558.data_ptr()), c_void_p(buf566.data_ptr()), c_void_p(buf574.data_ptr()), c_void_p(buf582.data_ptr()), c_void_p(buf590.data_ptr()), c_void_p(buf598.data_ptr()), c_void_p(buf606.data_ptr()), c_void_p(buf614.data_ptr()), c_void_p(buf622.data_ptr()), c_void_p(buf630.data_ptr()), c_void_p(buf638.data_ptr()), c_void_p(buf646.data_ptr()), c_void_p(buf654.data_ptr()), c_void_p(buf662.data_ptr()), c_void_p(buf670.data_ptr()), c_void_p(buf678.data_ptr()), c_void_p(buf686.data_ptr()), c_void_p(buf694.data_ptr()), c_void_p(buf702.data_ptr()), c_void_p(buf710.data_ptr()), c_void_p(buf718.data_ptr()), c_void_p(buf726.data_ptr()), c_void_p(buf734.data_ptr()), c_void_p(buf742.data_ptr()), c_void_p(buf750.data_ptr()), c_void_p(buf758.data_ptr()), c_void_p(buf766.data_ptr()), c_void_p(buf774.data_ptr()), c_void_p(buf782.data_ptr()), c_void_p(buf790.data_ptr()), c_void_p(buf798.data_ptr()), c_void_p(buf806.data_ptr()), c_void_p(buf814.data_ptr()), c_void_p(buf822.data_ptr()), c_void_p(buf830.data_ptr()), c_void_p(buf838.data_ptr()), c_void_p(buf846.data_ptr()), c_void_p(buf854.data_ptr()), c_void_p(buf862.data_ptr()), c_void_p(buf870.data_ptr()), c_void_p(buf878.data_ptr()), c_void_p(buf886.data_ptr()), c_void_p(buf894.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(primals_263.data_ptr()), c_void_p(primals_264.data_ptr()), c_void_p(primals_265.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(primals_266.data_ptr()), c_void_p(primals_267.data_ptr()), c_void_p(primals_268.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(primals_269.data_ptr()), c_void_p(primals_270.data_ptr()), c_void_p(primals_271.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(primals_272.data_ptr()), c_void_p(primals_273.data_ptr()), c_void_p(primals_274.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(primals_275.data_ptr()), c_void_p(primals_276.data_ptr()), c_void_p(primals_277.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(primals_278.data_ptr()), c_void_p(primals_279.data_ptr()), c_void_p(primals_280.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(primals_281.data_ptr()), c_void_p(primals_282.data_ptr()), c_void_p(primals_283.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(primals_284.data_ptr()), c_void_p(primals_285.data_ptr()), c_void_p(primals_286.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(primals_287.data_ptr()), c_void_p(primals_288.data_ptr()), c_void_p(primals_289.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(primals_290.data_ptr()), c_void_p(primals_291.data_ptr()), c_void_p(primals_292.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(primals_293.data_ptr()), c_void_p(primals_294.data_ptr()), c_void_p(primals_295.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(primals_296.data_ptr()), c_void_p(primals_297.data_ptr()), c_void_p(primals_298.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(primals_299.data_ptr()), c_void_p(primals_300.data_ptr()), c_void_p(primals_301.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(primals_302.data_ptr()), c_void_p(primals_303.data_ptr()), c_void_p(primals_304.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(primals_305.data_ptr()), c_void_p(primals_306.data_ptr()), c_void_p(primals_307.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(primals_308.data_ptr()), c_void_p(primals_309.data_ptr()), c_void_p(primals_310.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(primals_311.data_ptr()), c_void_p(primals_312.data_ptr()), c_void_p(primals_313.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(primals_314.data_ptr()), c_void_p(primals_315.data_ptr()), c_void_p(primals_316.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(primals_317.data_ptr()), c_void_p(primals_318.data_ptr()), c_void_p(primals_319.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(primals_320.data_ptr()), c_void_p(primals_321.data_ptr()), c_void_p(primals_322.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(primals_323.data_ptr()), c_void_p(primals_324.data_ptr()), c_void_p(primals_325.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(primals_326.data_ptr()), c_void_p(primals_327.data_ptr()), c_void_p(primals_328.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(primals_329.data_ptr()), c_void_p(primals_330.data_ptr()), c_void_p(primals_331.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(primals_332.data_ptr()), c_void_p(primals_333.data_ptr()), c_void_p(primals_334.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(primals_335.data_ptr()), c_void_p(primals_336.data_ptr()), c_void_p(primals_337.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(primals_338.data_ptr()), c_void_p(primals_339.data_ptr()), c_void_p(primals_340.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(primals_341.data_ptr()), c_void_p(primals_342.data_ptr()), c_void_p(primals_343.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(primals_344.data_ptr()), c_void_p(primals_345.data_ptr()), c_void_p(primals_346.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(primals_347.data_ptr()), c_void_p(primals_348.data_ptr()), c_void_p(primals_349.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(primals_350.data_ptr()), c_void_p(primals_351.data_ptr()), c_void_p(primals_352.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(primals_353.data_ptr()), c_void_p(primals_354.data_ptr()), c_void_p(primals_355.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(primals_356.data_ptr()), c_void_p(primals_357.data_ptr()), c_void_p(primals_358.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(primals_359.data_ptr()), c_void_p(primals_360.data_ptr()), c_void_p(primals_361.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(primals_362.data_ptr()), c_void_p(primals_363.data_ptr()), c_void_p(primals_364.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(primals_365.data_ptr()), c_void_p(primals_366.data_ptr()), c_void_p(primals_367.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(primals_368.data_ptr()), c_void_p(primals_369.data_ptr()), c_void_p(primals_370.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(primals_371.data_ptr()), c_void_p(primals_372.data_ptr()), c_void_p(primals_373.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(primals_374.data_ptr()), c_void_p(primals_375.data_ptr()), c_void_p(primals_376.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(primals_377.data_ptr()), c_void_p(primals_378.data_ptr()), c_void_p(primals_379.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(primals_380.data_ptr()), c_void_p(primals_381.data_ptr()), c_void_p(primals_382.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(primals_383.data_ptr()), c_void_p(primals_384.data_ptr()), c_void_p(primals_385.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(primals_386.data_ptr()), c_void_p(primals_387.data_ptr()), c_void_p(primals_388.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(primals_389.data_ptr()), c_void_p(primals_390.data_ptr()), c_void_p(primals_391.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(primals_392.data_ptr()), c_void_p(primals_393.data_ptr()), c_void_p(primals_394.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(primals_395.data_ptr()), c_void_p(primals_396.data_ptr()), c_void_p(primals_397.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(primals_398.data_ptr()), c_void_p(primals_399.data_ptr()), c_void_p(primals_400.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(primals_401.data_ptr()), c_void_p(primals_402.data_ptr()), c_void_p(primals_403.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(primals_404.data_ptr()), c_void_p(primals_405.data_ptr()), c_void_p(primals_406.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(primals_407.data_ptr()), c_void_p(primals_408.data_ptr()), c_void_p(primals_409.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(primals_410.data_ptr()), c_void_p(primals_411.data_ptr()), c_void_p(primals_412.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(primals_413.data_ptr()), c_void_p(primals_414.data_ptr()), c_void_p(primals_415.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(primals_416.data_ptr()), c_void_p(primals_417.data_ptr()), c_void_p(primals_418.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(primals_419.data_ptr()), c_void_p(primals_420.data_ptr()), c_void_p(primals_421.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(primals_422.data_ptr()), c_void_p(primals_423.data_ptr()), c_void_p(primals_424.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(primals_425.data_ptr()), c_void_p(primals_426.data_ptr()), c_void_p(primals_427.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(primals_428.data_ptr()), c_void_p(primals_429.data_ptr()), c_void_p(primals_430.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(primals_431.data_ptr()), c_void_p(primals_432.data_ptr()), c_void_p(primals_433.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(primals_434.data_ptr()), c_void_p(primals_435.data_ptr()), c_void_p(primals_436.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(primals_437.data_ptr()), c_void_p(primals_438.data_ptr()), c_void_p(primals_439.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(primals_440.data_ptr()), c_void_p(primals_441.data_ptr()), c_void_p(primals_442.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(primals_443.data_ptr()), c_void_p(primals_444.data_ptr()), c_void_p(primals_445.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(primals_446.data_ptr()), c_void_p(primals_447.data_ptr()), c_void_p(primals_448.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(primals_449.data_ptr()), c_void_p(primals_450.data_ptr()), c_void_p(primals_263.data_ptr()), c_void_p(primals_264.data_ptr()), c_void_p(primals_265.data_ptr()), c_void_p(primals_266.data_ptr()), c_void_p(primals_267.data_ptr()), c_void_p(primals_268.data_ptr()), c_void_p(primals_269.data_ptr()), c_void_p(primals_270.data_ptr()), c_void_p(primals_271.data_ptr()), c_void_p(primals_272.data_ptr()), c_void_p(primals_273.data_ptr()), c_void_p(primals_274.data_ptr()), c_void_p(primals_275.data_ptr()), c_void_p(primals_276.data_ptr()), c_void_p(primals_277.data_ptr()), c_void_p(primals_278.data_ptr()), c_void_p(primals_279.data_ptr()), c_void_p(primals_280.data_ptr()), c_void_p(primals_281.data_ptr()), c_void_p(primals_282.data_ptr()), c_void_p(primals_283.data_ptr()), c_void_p(primals_284.data_ptr()), c_void_p(primals_285.data_ptr()), c_void_p(primals_286.data_ptr()), c_void_p(primals_287.data_ptr()), c_void_p(primals_288.data_ptr()), c_void_p(primals_289.data_ptr()), c_void_p(primals_290.data_ptr()), c_void_p(primals_291.data_ptr()), c_void_p(primals_292.data_ptr()), c_void_p(primals_293.data_ptr()), c_void_p(primals_294.data_ptr()), c_void_p(primals_295.data_ptr()), c_void_p(primals_296.data_ptr()), c_void_p(primals_297.data_ptr()), c_void_p(primals_298.data_ptr()), c_void_p(primals_299.data_ptr()), c_void_p(primals_300.data_ptr()), c_void_p(primals_301.data_ptr()), c_void_p(primals_302.data_ptr()), c_void_p(primals_303.data_ptr()), c_void_p(primals_304.data_ptr()), c_void_p(primals_305.data_ptr()), c_void_p(primals_306.data_ptr()), c_void_p(primals_307.data_ptr()), c_void_p(primals_308.data_ptr()), c_void_p(primals_309.data_ptr()), c_void_p(primals_310.data_ptr()), c_void_p(primals_311.data_ptr()), c_void_p(primals_312.data_ptr()), c_void_p(primals_313.data_ptr()), c_void_p(primals_314.data_ptr()), c_void_p(primals_315.data_ptr()), c_void_p(primals_316.data_ptr()), c_void_p(primals_317.data_ptr()), c_void_p(primals_318.data_ptr()), c_void_p(primals_319.data_ptr()), c_void_p(primals_320.data_ptr()), c_void_p(primals_321.data_ptr()), c_void_p(primals_322.data_ptr()), c_void_p(primals_323.data_ptr()), c_void_p(primals_324.data_ptr()), c_void_p(primals_325.data_ptr()), c_void_p(primals_326.data_ptr()), c_void_p(primals_327.data_ptr()), c_void_p(primals_328.data_ptr()), c_void_p(primals_329.data_ptr()), c_void_p(primals_330.data_ptr()), c_void_p(primals_331.data_ptr()), c_void_p(primals_332.data_ptr()), c_void_p(primals_333.data_ptr()), c_void_p(primals_334.data_ptr()), c_void_p(primals_335.data_ptr()), c_void_p(primals_336.data_ptr()), c_void_p(primals_337.data_ptr()), c_void_p(primals_338.data_ptr()), c_void_p(primals_339.data_ptr()), c_void_p(primals_340.data_ptr()), c_void_p(primals_341.data_ptr()), c_void_p(primals_342.data_ptr()), c_void_p(primals_343.data_ptr()), c_void_p(primals_344.data_ptr()), c_void_p(primals_345.data_ptr()), c_void_p(primals_346.data_ptr()), c_void_p(primals_347.data_ptr()), c_void_p(primals_348.data_ptr()), c_void_p(primals_349.data_ptr()), c_void_p(primals_350.data_ptr()), c_void_p(primals_351.data_ptr()), c_void_p(primals_352.data_ptr()), c_void_p(primals_353.data_ptr()), c_void_p(primals_354.data_ptr()), c_void_p(primals_355.data_ptr()), c_void_p(primals_356.data_ptr()), c_void_p(primals_357.data_ptr()), c_void_p(primals_358.data_ptr()), c_void_p(primals_359.data_ptr()), c_void_p(primals_360.data_ptr()), c_void_p(primals_361.data_ptr()), c_void_p(primals_362.data_ptr()), c_void_p(primals_363.data_ptr()), c_void_p(primals_364.data_ptr()), c_void_p(primals_365.data_ptr()), c_void_p(primals_366.data_ptr()), c_void_p(primals_367.data_ptr()), c_void_p(primals_368.data_ptr()), c_void_p(primals_369.data_ptr()), c_void_p(primals_370.data_ptr()), c_void_p(primals_371.data_ptr()), c_void_p(primals_372.data_ptr()), c_void_p(primals_373.data_ptr()), c_void_p(primals_374.data_ptr()), c_void_p(primals_375.data_ptr()), c_void_p(primals_376.data_ptr()), c_void_p(primals_377.data_ptr()), c_void_p(primals_378.data_ptr()), c_void_p(primals_379.data_ptr()), c_void_p(primals_380.data_ptr()), c_void_p(primals_381.data_ptr()), c_void_p(primals_382.data_ptr()), c_void_p(primals_383.data_ptr()), c_void_p(primals_384.data_ptr()), c_void_p(primals_385.data_ptr()), c_void_p(primals_386.data_ptr()), c_void_p(primals_387.data_ptr()), c_void_p(primals_388.data_ptr()), c_void_p(primals_389.data_ptr()), c_void_p(primals_390.data_ptr()), c_void_p(primals_391.data_ptr()), c_void_p(primals_392.data_ptr()), c_void_p(primals_393.data_ptr()), c_void_p(primals_394.data_ptr()), c_void_p(primals_395.data_ptr()), c_void_p(primals_396.data_ptr()), c_void_p(primals_397.data_ptr()), c_void_p(primals_398.data_ptr()), c_void_p(primals_399.data_ptr()), c_void_p(primals_400.data_ptr()), c_void_p(primals_401.data_ptr()), c_void_p(primals_402.data_ptr()), c_void_p(primals_403.data_ptr()), c_void_p(primals_404.data_ptr()), c_void_p(primals_405.data_ptr()), c_void_p(primals_406.data_ptr()), c_void_p(primals_407.data_ptr()), c_void_p(primals_408.data_ptr()), c_void_p(primals_409.data_ptr()), c_void_p(primals_410.data_ptr()), c_void_p(primals_411.data_ptr()), c_void_p(primals_412.data_ptr()), c_void_p(primals_413.data_ptr()), c_void_p(primals_414.data_ptr()), c_void_p(primals_415.data_ptr()), c_void_p(primals_416.data_ptr()), c_void_p(primals_417.data_ptr()), c_void_p(primals_418.data_ptr()), c_void_p(primals_419.data_ptr()), c_void_p(primals_420.data_ptr()), c_void_p(primals_421.data_ptr()), c_void_p(primals_422.data_ptr()), c_void_p(primals_423.data_ptr()), c_void_p(primals_424.data_ptr()), c_void_p(primals_425.data_ptr()), c_void_p(primals_426.data_ptr()), c_void_p(primals_427.data_ptr()), c_void_p(primals_428.data_ptr()), c_void_p(primals_429.data_ptr()), c_void_p(primals_430.data_ptr()), c_void_p(primals_431.data_ptr()), c_void_p(primals_432.data_ptr()), c_void_p(primals_433.data_ptr()), c_void_p(primals_434.data_ptr()), c_void_p(primals_435.data_ptr()), c_void_p(primals_436.data_ptr()), c_void_p(primals_437.data_ptr()), c_void_p(primals_438.data_ptr()), c_void_p(primals_439.data_ptr()), c_void_p(primals_440.data_ptr()), c_void_p(primals_441.data_ptr()), c_void_p(primals_442.data_ptr()), c_void_p(primals_443.data_ptr()), c_void_p(primals_444.data_ptr()), c_void_p(primals_445.data_ptr()), c_void_p(primals_446.data_ptr()), c_void_p(primals_447.data_ptr()), c_void_p(primals_448.data_ptr()), c_void_p(primals_449.data_ptr()), c_void_p(primals_450.data_ptr()))
    del buf398
    del buf406
    del buf414
    del buf422
    del buf430
    del buf438
    del buf446
    del buf454
    del buf462
    del buf470
    del buf478
    del buf486
    del buf494
    del buf502
    del buf510
    del buf518
    del buf526
    del buf534
    del buf542
    del buf550
    del buf558
    del buf566
    del buf574
    del buf582
    del buf590
    del buf598
    del buf606
    del buf614
    del buf622
    del buf630
    del buf638
    del buf646
    del buf654
    del buf662
    del buf670
    del buf678
    del buf686
    del buf694
    del buf702
    del buf710
    del buf718
    del buf726
    del buf734
    del buf742
    del buf750
    del buf758
    del buf766
    del buf774
    del buf782
    del buf790
    del buf798
    del buf806
    del buf814
    del buf822
    del buf830
    del buf838
    del buf846
    del buf854
    del buf862
    del buf870
    del buf878
    del buf886
    del buf894
    del primals_263
    del primals_264
    del primals_265
    del primals_266
    del primals_267
    del primals_268
    del primals_269
    del primals_270
    del primals_271
    del primals_272
    del primals_273
    del primals_274
    del primals_275
    del primals_276
    del primals_277
    del primals_278
    del primals_279
    del primals_280
    del primals_281
    del primals_282
    del primals_283
    del primals_284
    del primals_285
    del primals_286
    del primals_287
    del primals_288
    del primals_289
    del primals_290
    del primals_291
    del primals_292
    del primals_293
    del primals_294
    del primals_295
    del primals_296
    del primals_297
    del primals_298
    del primals_299
    del primals_300
    del primals_301
    del primals_302
    del primals_303
    del primals_304
    del primals_305
    del primals_306
    del primals_307
    del primals_308
    del primals_309
    del primals_310
    del primals_311
    del primals_312
    del primals_313
    del primals_314
    del primals_315
    del primals_316
    del primals_317
    del primals_318
    del primals_319
    del primals_320
    del primals_321
    del primals_322
    del primals_323
    del primals_324
    del primals_325
    del primals_326
    del primals_327
    del primals_328
    del primals_329
    del primals_330
    del primals_331
    del primals_332
    del primals_333
    del primals_334
    del primals_335
    del primals_336
    del primals_337
    del primals_338
    del primals_339
    del primals_340
    del primals_341
    del primals_342
    del primals_343
    del primals_344
    del primals_345
    del primals_346
    del primals_347
    del primals_348
    del primals_349
    del primals_350
    del primals_351
    del primals_352
    del primals_353
    del primals_354
    del primals_355
    del primals_356
    del primals_357
    del primals_358
    del primals_359
    del primals_360
    del primals_361
    del primals_362
    del primals_363
    del primals_364
    del primals_365
    del primals_366
    del primals_367
    del primals_368
    del primals_369
    del primals_370
    del primals_371
    del primals_372
    del primals_373
    del primals_374
    del primals_375
    del primals_376
    del primals_377
    del primals_378
    del primals_379
    del primals_380
    del primals_381
    del primals_382
    del primals_383
    del primals_384
    del primals_385
    del primals_386
    del primals_387
    del primals_388
    del primals_389
    del primals_390
    del primals_391
    del primals_392
    del primals_393
    del primals_394
    del primals_395
    del primals_396
    del primals_397
    del primals_398
    del primals_399
    del primals_400
    del primals_401
    del primals_402
    del primals_403
    del primals_404
    del primals_405
    del primals_406
    del primals_407
    del primals_408
    del primals_409
    del primals_410
    del primals_411
    del primals_412
    del primals_413
    del primals_414
    del primals_415
    del primals_416
    del primals_417
    del primals_418
    del primals_419
    del primals_420
    del primals_421
    del primals_422
    del primals_423
    del primals_424
    del primals_425
    del primals_426
    del primals_427
    del primals_428
    del primals_429
    del primals_430
    del primals_431
    del primals_432
    del primals_433
    del primals_434
    del primals_435
    del primals_436
    del primals_437
    del primals_438
    del primals_439
    del primals_440
    del primals_441
    del primals_442
    del primals_443
    del primals_444
    del primals_445
    del primals_446
    del primals_447
    del primals_448
    del primals_449
    del primals_450
    buf902 = reinterpret_tensor(buf382, (768, ), (1, ), 0); del buf382  # reuse
    buf910 = reinterpret_tensor(buf388, (768, ), (1, ), 0); del buf388  # reuse
    cpp_fused__native_batch_norm_legit_functional_add_67(c_void_p(buf902.data_ptr()), c_void_p(buf910.data_ptr()), c_void_p(primals_451.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(primals_452.data_ptr()), c_void_p(primals_453.data_ptr()), c_void_p(primals_454.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(primals_455.data_ptr()), c_void_p(primals_456.data_ptr()), c_void_p(primals_457.data_ptr()), c_void_p(primals_451.data_ptr()), c_void_p(primals_452.data_ptr()), c_void_p(primals_453.data_ptr()), c_void_p(primals_454.data_ptr()), c_void_p(primals_455.data_ptr()), c_void_p(primals_456.data_ptr()), c_void_p(primals_457.data_ptr()))
    del buf902
    del buf910
    del primals_451
    del primals_452
    del primals_453
    del primals_454
    del primals_455
    del primals_456
    del primals_457
    return (buf393, buf0, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_137, primals_139, primals_141, primals_143, primals_145, primals_147, primals_149, primals_151, primals_153, primals_155, primals_157, primals_159, primals_161, primals_163, primals_165, primals_167, primals_169, primals_171, primals_173, primals_175, primals_177, primals_179, primals_181, primals_183, primals_185, primals_187, primals_189, primals_191, primals_193, primals_195, primals_197, primals_199, primals_201, primals_203, primals_205, primals_207, primals_209, primals_211, primals_213, primals_215, primals_217, primals_219, primals_221, primals_223, primals_225, primals_227, primals_229, primals_231, primals_233, primals_235, primals_237, primals_239, primals_241, primals_243, primals_245, primals_247, primals_249, primals_251, primals_253, primals_255, primals_257, primals_259, buf1, buf2, buf6, buf7, buf8, buf12, buf13, buf14, buf18, buf19, buf20, buf24, buf25, buf26, buf30, buf31, buf32, buf36, buf37, buf38, buf42, buf43, buf44, buf48, buf49, buf50, buf54, buf55, buf56, buf60, buf61, buf62, buf66, buf67, buf68, buf72, buf73, buf74, buf78, buf79, buf80, buf84, buf85, buf86, buf90, buf91, buf92, buf96, buf97, buf98, buf102, buf103, buf104, buf108, buf109, buf110, buf114, buf115, buf116, buf120, buf121, buf122, buf126, buf127, buf128, buf132, buf133, buf134, buf138, buf139, buf140, buf144, buf145, buf146, buf150, buf151, buf152, buf156, buf157, buf158, buf162, buf163, buf164, buf168, buf169, buf170, buf174, buf175, buf176, buf180, buf181, buf182, buf186, buf187, buf188, buf192, buf193, buf194, buf198, buf199, buf200, buf204, buf205, buf206, buf210, buf211, buf212, buf216, buf217, buf218, buf222, buf223, buf224, buf228, buf229, buf230, buf234, buf235, buf236, buf240, buf241, buf242, buf246, buf247, buf248, buf252, buf253, buf254, buf258, buf259, buf260, buf264, buf265, buf266, buf270, buf271, buf272, buf276, buf277, buf278, buf282, buf283, buf284, buf288, buf289, buf290, buf294, buf295, buf296, buf300, buf301, buf302, buf306, buf307, buf308, buf312, buf313, buf314, buf318, buf319, buf320, buf324, buf325, buf326, buf330, buf331, buf332, buf336, buf337, buf338, buf342, buf343, buf344, buf348, buf349, buf350, buf354, buf355, buf356, buf360, buf361, buf362, buf366, buf367, buf368, buf372, buf373, buf374, buf378, buf379, buf380, buf384, buf385, buf386, buf390, buf392, reinterpret_tensor(primals_261, (1000, 768), (768, 1), 0), reinterpret_tensor(buf387, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf381, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf375, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf369, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf363, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf357, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf351, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf345, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf339, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf333, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf327, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf321, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf315, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf309, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf303, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf297, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf291, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf285, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf279, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf273, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf267, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf261, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf255, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf249, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf243, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf237, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf231, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf225, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf219, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf213, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf207, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf201, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf195, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf189, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf183, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf177, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf171, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf165, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf159, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf153, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf147, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf141, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf135, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf129, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf123, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf117, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf111, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf105, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf99, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf93, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf87, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf81, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf75, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf69, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf63, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf57, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf51, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf45, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf39, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf33, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf27, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf21, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf15, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf9, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf3, (1, 768, 1, 1), (768, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((768, 3, 7, 7), (147, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_204 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_206 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_207 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_209 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_210 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_212 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_213 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_214 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_215 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_216 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_218 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_219 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_220 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_221 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_222 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_223 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_224 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_225 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_226 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_227 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_228 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_229 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_230 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_231 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_232 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_233 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_234 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_235 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_236 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_237 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_238 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_239 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_240 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_241 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_242 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_243 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_244 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_245 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_246 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_247 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_248 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_249 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_250 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_251 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_252 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_253 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_254 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_255 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_256 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_257 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_258 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_259 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_260 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_261 = rand_strided((1000, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_262 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    primals_263 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_264 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_265 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_266 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_267 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_268 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_269 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_270 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_271 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_272 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_273 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_274 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_275 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_276 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_277 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_278 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_279 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_280 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_281 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_282 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_283 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_284 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_285 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_286 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_287 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_288 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_289 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_290 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_291 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_292 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_293 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_294 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_295 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_296 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_297 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_298 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_299 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_300 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_301 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_302 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_303 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_304 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_305 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_306 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_307 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_308 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_309 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_310 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_311 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_312 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_313 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_314 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_315 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_316 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_317 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_318 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_319 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_320 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_321 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_322 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_323 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_324 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_325 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_326 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_327 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_328 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_329 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_330 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_331 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_332 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_333 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_334 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_335 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_336 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_337 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_338 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_339 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_340 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_341 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_342 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_343 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_344 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_345 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_346 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_347 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_348 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_349 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_350 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_351 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_352 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_353 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_354 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_355 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_356 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_357 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_358 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_359 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_360 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_361 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_362 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_363 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_364 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_365 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_366 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_367 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_368 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_369 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_370 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_371 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_372 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_373 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_374 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_375 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_376 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_377 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_378 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_379 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_380 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_381 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_382 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_383 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_384 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_385 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_386 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_387 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_388 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_389 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_390 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_391 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_392 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_393 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_394 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_395 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_396 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_397 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_398 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_399 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_400 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_401 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_402 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_403 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_404 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_405 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_406 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_407 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_408 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_409 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_410 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_411 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_412 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_413 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_414 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_415 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_416 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_417 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_418 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_419 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_420 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_421 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_422 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_423 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_424 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_425 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_426 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_427 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_428 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_429 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_430 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_431 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_432 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_433 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_434 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_435 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_436 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_437 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_438 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_439 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_440 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_441 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_442 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_443 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_444 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_445 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_446 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_447 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_448 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_449 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_450 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_451 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_452 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_453 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_454 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_455 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_456 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_457 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_458 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('convmixer_768_32', benchmark_compiled_module)
