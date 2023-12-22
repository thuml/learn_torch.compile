
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x2 + (9L*x1) + (27L*x0))];
                    out_ptr0[static_cast<long>(x1 + (3L*x2) + (27L*x0))] = tmp0;
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


cpp_fused__native_batch_norm_legit_functional_silu_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (32L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(100352.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(100352.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_hardtanh_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (32L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(100352.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(100352.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = at::vec::maximum(tmp0, tmp2);
                auto tmp4 = static_cast<float>(6.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::minimum(tmp3, tmp5);
                tmp6.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_3 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (16L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(100352.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(100352.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_silu_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (96L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(100352.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(100352.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9633792L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_hardtanh_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (96L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(25088.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(25088.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = at::vec::maximum(tmp0, tmp2);
                auto tmp4 = static_cast<float>(6.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::minimum(tmp3, tmp5);
                tmp6.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_6 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (27L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(24L); x0<static_cast<long>(27L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (27L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(24L); x0<static_cast<long>(27L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (27L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(25088.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (27L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(24L); x1<static_cast<long>(27L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (27L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(25088.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    out_ptr3[static_cast<long>(x1 + (27L*x0))] = tmp13;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_silu_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (162L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(160L); x0<static_cast<long>(162L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (162L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(160L); x0<static_cast<long>(162L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (162L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(25088.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (162L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(160L); x1<static_cast<long>(162L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (162L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(25088.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    out_ptr3[static_cast<long>(x1 + (162L*x0))] = tmp13;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4064256L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_hardtanh_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (162L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(160L); x0<static_cast<long>(162L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (162L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(160L); x0<static_cast<long>(162L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (162L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(25088.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (162L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(160L); x1<static_cast<long>(162L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (162L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(25088.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    out_ptr3[static_cast<long>(x1 + (162L*x0))] = tmp13;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4064256L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = at::vec::maximum(tmp0, tmp2);
                auto tmp4 = static_cast<float>(6.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::minimum(tmp3, tmp5);
                tmp6.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_cat_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (38L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(32L); x0<static_cast<long>(38L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (38L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(32L); x0<static_cast<long>(38L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(38L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (38L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(25088.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    auto tmp14 = c10::convert<long>(x1);
                    auto tmp15 = static_cast<long>(0);
                    auto tmp16 = tmp14 >= tmp15;
                    auto tmp17 = static_cast<long>(27);
                    auto tmp18 = tmp14 < tmp17;
                    auto tmp19 = [&]
                    {
                        auto tmp20 = in_ptr3[static_cast<long>(x1 + (27L*x0))];
                        auto tmp21 = decltype(tmp13)(tmp13 + tmp20);
                        return tmp21;
                    }
                    ;
                    auto tmp22 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                    auto tmp23 = tmp14 >= tmp17;
                    auto tmp24 = static_cast<long>(38);
                    auto tmp25 = tmp14 < tmp24;
                    auto tmp26 = [&]
                    {
                        return tmp13;
                    }
                    ;
                    auto tmp27 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                    auto tmp28 = tmp18 ? tmp22 : tmp27;
                    in_out_ptr0[static_cast<long>(x1 + (38L*x0))] = tmp28;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_silu_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (228L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(224L); x0<static_cast<long>(228L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (228L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(224L); x0<static_cast<long>(228L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (228L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(25088.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (228L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(224L); x1<static_cast<long>(228L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (228L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(25088.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    out_ptr3[static_cast<long>(x1 + (228L*x0))] = tmp13;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(5720064L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_mean_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (228L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(224L); x0<static_cast<long>(228L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (228L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(224L); x0<static_cast<long>(228L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (228L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(6272.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (228L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(224L); x1<static_cast<long>(228L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (228L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(6272.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    out_ptr3[static_cast<long>(x1 + (228L*x0))] = tmp13;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1 + (228L*x2) + (178752L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x1 + (228L*x0)));
                    }
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(224L); x1<static_cast<long>(228L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr3[static_cast<long>(x1 + (228L*x2) + (178752L*x0))];
                            tmp_acc0 = tmp_acc0 + tmp0;
                        }
                        out_ptr4[static_cast<long>(x1 + (228L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1824L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(784.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (19L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(16L); x0<static_cast<long>(19L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                Welford<float> tmp_acc0 = Welford<float>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (19L*x1))];
                    tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (19L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(8.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp11 = tmp2 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                tmp16.store(out_ptr2 + static_cast<long>(x1 + (19L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(16L); x1<static_cast<long>(19L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (19L*x0))];
                auto tmp1 = out_ptr0[static_cast<long>(x1)];
                auto tmp3 = out_ptr1[static_cast<long>(x1)];
                auto tmp10 = in_ptr1[static_cast<long>(x1)];
                auto tmp12 = in_ptr2[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(8.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr2[static_cast<long>(x1 + (19L*x0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused_hardtanh_mul_sigmoid_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(224L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (228L*x1) + (178752L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (228L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = at::vec::maximum(tmp3, tmp5);
                        auto tmp7 = static_cast<float>(6.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = at::vec::minimum(tmp6, tmp8);
                        tmp9.store(out_ptr0 + static_cast<long>(x2 + (228L*x1) + (178752L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(224L); x2<static_cast<long>(228L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (228L*x1) + (178752L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (228L*x0))];
                        auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                        auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = max_propagate_nan(tmp3, tmp4);
                        auto tmp6 = static_cast<float>(6.0);
                        auto tmp7 = min_propagate_nan(tmp5, tmp6);
                        out_ptr0[static_cast<long>(x2 + (228L*x1) + (178752L*x0))] = tmp7;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_14 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (50L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(48L); x0<static_cast<long>(50L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (50L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(48L); x0<static_cast<long>(50L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (50L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(6272.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (50L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(50L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (50L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(6272.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    out_ptr3[static_cast<long>(x1 + (50L*x0))] = tmp13;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_silu_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(296L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (300L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(296L); x0<static_cast<long>(300L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (300L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(296L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(296L); x0<static_cast<long>(300L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(296L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (300L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(6272.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (300L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(296L); x1<static_cast<long>(300L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (300L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(6272.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    out_ptr3[static_cast<long>(x1 + (300L*x0))] = tmp13;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1881600L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_mean_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(296L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (300L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(296L); x0<static_cast<long>(300L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (300L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(296L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(296L); x0<static_cast<long>(300L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(296L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (300L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(6272.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (300L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(296L); x1<static_cast<long>(300L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (300L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(6272.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    out_ptr3[static_cast<long>(x1 + (300L*x0))] = tmp13;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(296L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1 + (300L*x2) + (235200L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x1 + (300L*x0)));
                    }
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(296L); x1<static_cast<long>(300L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr3[static_cast<long>(x1 + (300L*x2) + (235200L*x0))];
                            tmp_acc0 = tmp_acc0 + tmp0;
                        }
                        out_ptr4[static_cast<long>(x1 + (300L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2400L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(784.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
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
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (25L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(24L); x0<static_cast<long>(25L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                Welford<float> tmp_acc0 = Welford<float>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (25L*x1))];
                    tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (25L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(8.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp11 = tmp2 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                tmp16.store(out_ptr2 + static_cast<long>(x1 + (25L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(24L); x1<static_cast<long>(25L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (25L*x0))];
                auto tmp1 = out_ptr0[static_cast<long>(x1)];
                auto tmp3 = out_ptr1[static_cast<long>(x1)];
                auto tmp10 = in_ptr1[static_cast<long>(x1)];
                auto tmp12 = in_ptr2[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(8.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr2[static_cast<long>(x1 + (25L*x0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused_hardtanh_mul_sigmoid_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(296L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (300L*x1) + (235200L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (300L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = at::vec::maximum(tmp3, tmp5);
                        auto tmp7 = static_cast<float>(6.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = at::vec::minimum(tmp6, tmp8);
                        tmp9.store(out_ptr0 + static_cast<long>(x2 + (300L*x1) + (235200L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(296L); x2<static_cast<long>(300L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (300L*x1) + (235200L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (300L*x0))];
                        auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                        auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = max_propagate_nan(tmp3, tmp4);
                        auto tmp6 = static_cast<float>(6.0);
                        auto tmp7 = min_propagate_nan(tmp5, tmp6);
                        out_ptr0[static_cast<long>(x2 + (300L*x1) + (235200L*x0))] = tmp7;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_cat_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (61L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(56L); x0<static_cast<long>(61L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (61L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(56L); x0<static_cast<long>(61L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(61L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (61L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(6272.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    auto tmp14 = c10::convert<long>(x1);
                    auto tmp15 = static_cast<long>(0);
                    auto tmp16 = tmp14 >= tmp15;
                    auto tmp17 = static_cast<long>(50);
                    auto tmp18 = tmp14 < tmp17;
                    auto tmp19 = [&]
                    {
                        auto tmp20 = in_ptr3[static_cast<long>(x1 + (50L*x0))];
                        auto tmp21 = decltype(tmp13)(tmp13 + tmp20);
                        return tmp21;
                    }
                    ;
                    auto tmp22 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                    auto tmp23 = tmp14 >= tmp17;
                    auto tmp24 = static_cast<long>(61);
                    auto tmp25 = tmp14 < tmp24;
                    auto tmp26 = [&]
                    {
                        return tmp13;
                    }
                    ;
                    auto tmp27 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                    auto tmp28 = tmp18 ? tmp22 : tmp27;
                    in_out_ptr0[static_cast<long>(x1 + (61L*x0))] = tmp28;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_silu_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(360L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (366L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(360L); x0<static_cast<long>(366L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (366L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(360L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(360L); x0<static_cast<long>(366L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (366L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(6272.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (366L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(360L); x1<static_cast<long>(366L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (366L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(6272.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    out_ptr3[static_cast<long>(x1 + (366L*x0))] = tmp13;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2295552L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_mean_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(360L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (366L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(360L); x0<static_cast<long>(366L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (366L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(360L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(360L); x0<static_cast<long>(366L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (366L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (366L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(360L); x1<static_cast<long>(366L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (366L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    out_ptr3[static_cast<long>(x1 + (366L*x0))] = tmp13;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1 + (366L*x2) + (71736L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x1 + (366L*x0)));
                    }
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(360L); x1<static_cast<long>(366L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr3[static_cast<long>(x1 + (366L*x2) + (71736L*x0))];
                            tmp_acc0 = tmp_acc0 + tmp0;
                        }
                        out_ptr4[static_cast<long>(x1 + (366L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2928L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(196.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (30L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(24L); x0<static_cast<long>(30L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                Welford<float> tmp_acc0 = Welford<float>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (30L*x1))];
                    tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (30L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(8.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp11 = tmp2 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                tmp16.store(out_ptr2 + static_cast<long>(x1 + (30L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(24L); x1<static_cast<long>(30L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (30L*x0))];
                auto tmp1 = out_ptr0[static_cast<long>(x1)];
                auto tmp3 = out_ptr1[static_cast<long>(x1)];
                auto tmp10 = in_ptr1[static_cast<long>(x1)];
                auto tmp12 = in_ptr2[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(8.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr2[static_cast<long>(x1 + (30L*x0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused_hardtanh_mul_sigmoid_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(360L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (366L*x1) + (71736L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (366L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = at::vec::maximum(tmp3, tmp5);
                        auto tmp7 = static_cast<float>(6.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = at::vec::minimum(tmp6, tmp8);
                        tmp9.store(out_ptr0 + static_cast<long>(x2 + (366L*x1) + (71736L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(360L); x2<static_cast<long>(366L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (366L*x1) + (71736L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (366L*x0))];
                        auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                        auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = max_propagate_nan(tmp3, tmp4);
                        auto tmp6 = static_cast<float>(6.0);
                        auto tmp7 = min_propagate_nan(tmp5, tmp6);
                        out_ptr0[static_cast<long>(x2 + (366L*x1) + (71736L*x0))] = tmp7;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (72L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(1568.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (72L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1568.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp11 = tmp2 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr3 + static_cast<long>(x1 + (72L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_silu_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(432L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (432L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(432L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(432L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (432L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (432L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(677376L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_mean_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(432L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (432L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(432L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(432L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (432L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (432L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(432L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1 + (432L*x2) + (84672L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x1 + (432L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3456L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(196.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
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
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (36L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(32L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                Welford<float> tmp_acc0 = Welford<float>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (36L*x1))];
                    tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (36L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(8.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp11 = tmp2 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                tmp16.store(out_ptr2 + static_cast<long>(x1 + (36L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(32L); x1<static_cast<long>(36L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (36L*x0))];
                auto tmp1 = out_ptr0[static_cast<long>(x1)];
                auto tmp3 = out_ptr1[static_cast<long>(x1)];
                auto tmp10 = in_ptr1[static_cast<long>(x1)];
                auto tmp12 = in_ptr2[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(8.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr2[static_cast<long>(x1 + (36L*x0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused_hardtanh_mul_sigmoid_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(432L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (432L*x1) + (84672L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (432L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = at::vec::maximum(tmp3, tmp5);
                        auto tmp7 = static_cast<float>(6.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = at::vec::minimum(tmp6, tmp8);
                        tmp9.store(out_ptr0 + static_cast<long>(x2 + (432L*x1) + (84672L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_cat_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (84L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(80L); x0<static_cast<long>(84L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (84L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(80L); x0<static_cast<long>(84L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(84L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (84L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    auto tmp14 = c10::convert<long>(x1);
                    auto tmp15 = static_cast<long>(0);
                    auto tmp16 = tmp14 >= tmp15;
                    auto tmp17 = static_cast<long>(72);
                    auto tmp18 = tmp14 < tmp17;
                    auto tmp19 = [&]
                    {
                        auto tmp20 = in_ptr3[static_cast<long>(x1 + (72L*x0))];
                        auto tmp21 = decltype(tmp13)(tmp13 + tmp20);
                        return tmp21;
                    }
                    ;
                    auto tmp22 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                    auto tmp23 = tmp14 >= tmp17;
                    auto tmp24 = static_cast<long>(84);
                    auto tmp25 = tmp14 < tmp24;
                    auto tmp26 = [&]
                    {
                        return tmp13;
                    }
                    ;
                    auto tmp27 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                    auto tmp28 = tmp18 ? tmp22 : tmp27;
                    in_out_ptr0[static_cast<long>(x1 + (84L*x0))] = tmp28;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_silu_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(504L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (504L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(504L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(504L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (504L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (504L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(790272L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_mean_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(504L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (504L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(504L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(504L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (504L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (504L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(504L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1 + (504L*x2) + (98784L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x1 + (504L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4032L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(196.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (42L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(40L); x0<static_cast<long>(42L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                Welford<float> tmp_acc0 = Welford<float>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (42L*x1))];
                    tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (42L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(8.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp11 = tmp2 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                tmp16.store(out_ptr2 + static_cast<long>(x1 + (42L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(40L); x1<static_cast<long>(42L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (42L*x0))];
                auto tmp1 = out_ptr0[static_cast<long>(x1)];
                auto tmp3 = out_ptr1[static_cast<long>(x1)];
                auto tmp10 = in_ptr1[static_cast<long>(x1)];
                auto tmp12 = in_ptr2[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(8.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr2[static_cast<long>(x1 + (42L*x0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused_hardtanh_mul_sigmoid_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(504L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (504L*x1) + (98784L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (504L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = at::vec::maximum(tmp3, tmp5);
                        auto tmp7 = static_cast<float>(6.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = at::vec::minimum(tmp6, tmp8);
                        tmp9.store(out_ptr0 + static_cast<long>(x2 + (504L*x1) + (98784L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_cat_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(88L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (95L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(88L); x0<static_cast<long>(95L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (95L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(88L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(88L); x0<static_cast<long>(95L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(95L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (95L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    auto tmp14 = c10::convert<long>(x1);
                    auto tmp15 = static_cast<long>(0);
                    auto tmp16 = tmp14 >= tmp15;
                    auto tmp17 = static_cast<long>(84);
                    auto tmp18 = tmp14 < tmp17;
                    auto tmp19 = [&]
                    {
                        auto tmp20 = in_ptr3[static_cast<long>(x1 + (84L*x0))];
                        auto tmp21 = decltype(tmp13)(tmp13 + tmp20);
                        return tmp21;
                    }
                    ;
                    auto tmp22 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                    auto tmp23 = tmp14 >= tmp17;
                    auto tmp24 = static_cast<long>(95);
                    auto tmp25 = tmp14 < tmp24;
                    auto tmp26 = [&]
                    {
                        return tmp13;
                    }
                    ;
                    auto tmp27 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                    auto tmp28 = tmp18 ? tmp22 : tmp27;
                    in_out_ptr0[static_cast<long>(x1 + (95L*x0))] = tmp28;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_silu_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(568L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (570L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(568L); x0<static_cast<long>(570L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (570L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(568L); x0<static_cast<long>(570L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(568L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (570L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (570L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(568L); x1<static_cast<long>(570L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (570L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    out_ptr3[static_cast<long>(x1 + (570L*x0))] = tmp13;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(893760L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_mean_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(568L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (570L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(568L); x0<static_cast<long>(570L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (570L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(568L); x0<static_cast<long>(570L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(568L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (570L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (570L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(568L); x1<static_cast<long>(570L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (570L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    out_ptr3[static_cast<long>(x1 + (570L*x0))] = tmp13;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(568L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1 + (570L*x2) + (111720L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x1 + (570L*x0)));
                    }
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(568L); x1<static_cast<long>(570L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr3[static_cast<long>(x1 + (570L*x2) + (111720L*x0))];
                            tmp_acc0 = tmp_acc0 + tmp0;
                        }
                        out_ptr4[static_cast<long>(x1 + (570L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4560L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(196.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
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
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (47L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(40L); x0<static_cast<long>(47L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                Welford<float> tmp_acc0 = Welford<float>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (47L*x1))];
                    tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (47L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(8.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp11 = tmp2 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                tmp16.store(out_ptr2 + static_cast<long>(x1 + (47L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(40L); x1<static_cast<long>(47L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (47L*x0))];
                auto tmp1 = out_ptr0[static_cast<long>(x1)];
                auto tmp3 = out_ptr1[static_cast<long>(x1)];
                auto tmp10 = in_ptr1[static_cast<long>(x1)];
                auto tmp12 = in_ptr2[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(8.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr2[static_cast<long>(x1 + (47L*x0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused_hardtanh_mul_sigmoid_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(568L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (570L*x1) + (111720L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (570L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = at::vec::maximum(tmp3, tmp5);
                        auto tmp7 = static_cast<float>(6.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = at::vec::minimum(tmp6, tmp8);
                        tmp9.store(out_ptr0 + static_cast<long>(x2 + (570L*x1) + (111720L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(568L); x2<static_cast<long>(570L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (570L*x1) + (111720L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (570L*x0))];
                        auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                        auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = max_propagate_nan(tmp3, tmp4);
                        auto tmp6 = static_cast<float>(6.0);
                        auto tmp7 = min_propagate_nan(tmp5, tmp6);
                        out_ptr0[static_cast<long>(x2 + (570L*x1) + (111720L*x0))] = tmp7;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_cat_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(104L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (106L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(104L); x0<static_cast<long>(106L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (106L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(104L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(104L); x0<static_cast<long>(106L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(106L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (106L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    auto tmp14 = c10::convert<long>(x1);
                    auto tmp15 = static_cast<long>(0);
                    auto tmp16 = tmp14 >= tmp15;
                    auto tmp17 = static_cast<long>(95);
                    auto tmp18 = tmp14 < tmp17;
                    auto tmp19 = [&]
                    {
                        auto tmp20 = in_ptr3[static_cast<long>(x1 + (95L*x0))];
                        auto tmp21 = decltype(tmp13)(tmp13 + tmp20);
                        return tmp21;
                    }
                    ;
                    auto tmp22 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                    auto tmp23 = tmp14 >= tmp17;
                    auto tmp24 = static_cast<long>(106);
                    auto tmp25 = tmp14 < tmp24;
                    auto tmp26 = [&]
                    {
                        return tmp13;
                    }
                    ;
                    auto tmp27 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                    auto tmp28 = tmp18 ? tmp22 : tmp27;
                    in_out_ptr0[static_cast<long>(x1 + (106L*x0))] = tmp28;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_silu_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(632L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (636L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(632L); x0<static_cast<long>(636L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (636L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(632L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(632L); x0<static_cast<long>(636L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(632L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (636L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (636L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(632L); x1<static_cast<long>(636L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (636L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    out_ptr3[static_cast<long>(x1 + (636L*x0))] = tmp13;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(997248L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_mean_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(632L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (636L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(632L); x0<static_cast<long>(636L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (636L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(632L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(632L); x0<static_cast<long>(636L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(632L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (636L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (636L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(632L); x1<static_cast<long>(636L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (636L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    out_ptr3[static_cast<long>(x1 + (636L*x0))] = tmp13;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(632L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1 + (636L*x2) + (124656L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x1 + (636L*x0)));
                    }
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(632L); x1<static_cast<long>(636L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr3[static_cast<long>(x1 + (636L*x2) + (124656L*x0))];
                            tmp_acc0 = tmp_acc0 + tmp0;
                        }
                        out_ptr4[static_cast<long>(x1 + (636L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5088L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(196.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (53L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(48L); x0<static_cast<long>(53L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                Welford<float> tmp_acc0 = Welford<float>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (53L*x1))];
                    tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (53L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(8.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp11 = tmp2 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                tmp16.store(out_ptr2 + static_cast<long>(x1 + (53L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(48L); x1<static_cast<long>(53L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (53L*x0))];
                auto tmp1 = out_ptr0[static_cast<long>(x1)];
                auto tmp3 = out_ptr1[static_cast<long>(x1)];
                auto tmp10 = in_ptr1[static_cast<long>(x1)];
                auto tmp12 = in_ptr2[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(8.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr2[static_cast<long>(x1 + (53L*x0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused_hardtanh_mul_sigmoid_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(632L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (636L*x1) + (124656L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (636L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = at::vec::maximum(tmp3, tmp5);
                        auto tmp7 = static_cast<float>(6.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = at::vec::minimum(tmp6, tmp8);
                        tmp9.store(out_ptr0 + static_cast<long>(x2 + (636L*x1) + (124656L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(632L); x2<static_cast<long>(636L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (636L*x1) + (124656L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (636L*x0))];
                        auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                        auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = max_propagate_nan(tmp3, tmp4);
                        auto tmp6 = static_cast<float>(6.0);
                        auto tmp7 = min_propagate_nan(tmp5, tmp6);
                        out_ptr0[static_cast<long>(x2 + (636L*x1) + (124656L*x0))] = tmp7;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_cat_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (117L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(112L); x0<static_cast<long>(117L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (117L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(112L); x0<static_cast<long>(117L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(117L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (117L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    auto tmp14 = c10::convert<long>(x1);
                    auto tmp15 = static_cast<long>(0);
                    auto tmp16 = tmp14 >= tmp15;
                    auto tmp17 = static_cast<long>(106);
                    auto tmp18 = tmp14 < tmp17;
                    auto tmp19 = [&]
                    {
                        auto tmp20 = in_ptr3[static_cast<long>(x1 + (106L*x0))];
                        auto tmp21 = decltype(tmp13)(tmp13 + tmp20);
                        return tmp21;
                    }
                    ;
                    auto tmp22 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                    auto tmp23 = tmp14 >= tmp17;
                    auto tmp24 = static_cast<long>(117);
                    auto tmp25 = tmp14 < tmp24;
                    auto tmp26 = [&]
                    {
                        return tmp13;
                    }
                    ;
                    auto tmp27 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                    auto tmp28 = tmp18 ? tmp22 : tmp27;
                    in_out_ptr0[static_cast<long>(x1 + (117L*x0))] = tmp28;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_silu_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(696L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (702L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(696L); x0<static_cast<long>(702L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (702L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(696L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(696L); x0<static_cast<long>(702L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(696L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (702L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (702L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(696L); x1<static_cast<long>(702L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (702L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    out_ptr3[static_cast<long>(x1 + (702L*x0))] = tmp13;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1100736L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_mean_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(696L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (702L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(696L); x0<static_cast<long>(702L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (702L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(696L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(696L); x0<static_cast<long>(702L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(696L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (702L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (702L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(696L); x1<static_cast<long>(702L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (702L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    out_ptr3[static_cast<long>(x1 + (702L*x0))] = tmp13;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(696L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1 + (702L*x2) + (137592L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x1 + (702L*x0)));
                    }
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(696L); x1<static_cast<long>(702L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr3[static_cast<long>(x1 + (702L*x2) + (137592L*x0))];
                            tmp_acc0 = tmp_acc0 + tmp0;
                        }
                        out_ptr4[static_cast<long>(x1 + (702L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5616L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(196.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
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
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (58L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(56L); x0<static_cast<long>(58L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                Welford<float> tmp_acc0 = Welford<float>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (58L*x1))];
                    tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (58L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(8.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp11 = tmp2 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                tmp16.store(out_ptr2 + static_cast<long>(x1 + (58L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(56L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (58L*x0))];
                auto tmp1 = out_ptr0[static_cast<long>(x1)];
                auto tmp3 = out_ptr1[static_cast<long>(x1)];
                auto tmp10 = in_ptr1[static_cast<long>(x1)];
                auto tmp12 = in_ptr2[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(8.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr2[static_cast<long>(x1 + (58L*x0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused_hardtanh_mul_sigmoid_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(696L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (702L*x1) + (137592L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (702L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = at::vec::maximum(tmp3, tmp5);
                        auto tmp7 = static_cast<float>(6.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = at::vec::minimum(tmp6, tmp8);
                        tmp9.store(out_ptr0 + static_cast<long>(x2 + (702L*x1) + (137592L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(696L); x2<static_cast<long>(702L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (702L*x1) + (137592L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (702L*x0))];
                        auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                        auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = max_propagate_nan(tmp3, tmp4);
                        auto tmp6 = static_cast<float>(6.0);
                        auto tmp7 = min_propagate_nan(tmp5, tmp6);
                        out_ptr0[static_cast<long>(x2 + (702L*x1) + (137592L*x0))] = tmp7;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_cat_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    auto tmp14 = c10::convert<long>(x1);
                    auto tmp15 = static_cast<long>(0);
                    auto tmp16 = tmp14 >= tmp15;
                    auto tmp17 = static_cast<long>(117);
                    auto tmp18 = tmp14 < tmp17;
                    auto tmp19 = [&]
                    {
                        auto tmp20 = in_ptr3[static_cast<long>(x1 + (117L*x0))];
                        auto tmp21 = decltype(tmp13)(tmp13 + tmp20);
                        return tmp21;
                    }
                    ;
                    auto tmp22 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                    auto tmp23 = tmp14 >= tmp17;
                    auto tmp24 = static_cast<long>(128);
                    auto tmp25 = tmp14 < tmp24;
                    auto tmp26 = [&]
                    {
                        return tmp13;
                    }
                    ;
                    auto tmp27 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                    auto tmp28 = tmp18 ? tmp22 : tmp27;
                    in_out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp28;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_silu_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
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
                    auto tmp1 = static_cast<float>(1568.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1204224L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_mean_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
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
                    auto tmp1 = static_cast<float>(392.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(392.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1 + (768L*x2) + (37632L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(49.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(8.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp11 = tmp2 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                tmp16.store(out_ptr2 + static_cast<long>(x1 + (64L*x0)));
            }
        }
    }
}
''')


cpp_fused_hardtanh_mul_sigmoid_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (37632L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = at::vec::maximum(tmp3, tmp5);
                        auto tmp7 = static_cast<float>(6.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = at::vec::minimum(tmp6, tmp8);
                        tmp9.store(out_ptr0 + static_cast<long>(x2 + (768L*x1) + (37632L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(136L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (140L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(136L); x0<static_cast<long>(140L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                Welford<float> tmp_acc0 = Welford<float>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (140L*x1))];
                    tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(136L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(136L); x0<static_cast<long>(140L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
            auto tmp5 = 1 / std::sqrt(tmp4);
            out_ptr2[static_cast<long>(x0)] = tmp5;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(136L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (140L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(392.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp11 = tmp2 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr3 + static_cast<long>(x1 + (140L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(136L); x1<static_cast<long>(140L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (140L*x0))];
                auto tmp1 = out_ptr0[static_cast<long>(x1)];
                auto tmp3 = out_ptr1[static_cast<long>(x1)];
                auto tmp10 = in_ptr1[static_cast<long>(x1)];
                auto tmp12 = in_ptr2[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(392.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                out_ptr3[static_cast<long>(x1 + (140L*x0))] = tmp13;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_silu_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(840L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (840L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(840L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(840L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (840L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(392.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (840L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(329280L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_mean_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(840L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (840L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(840L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(840L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (840L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(392.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (840L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(840L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1 + (840L*x2) + (41160L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x1 + (840L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(6720L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(49.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
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
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (70L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(64L); x0<static_cast<long>(70L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                Welford<float> tmp_acc0 = Welford<float>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (70L*x1))];
                    tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (70L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(8.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp11 = tmp2 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                tmp16.store(out_ptr2 + static_cast<long>(x1 + (70L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(64L); x1<static_cast<long>(70L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (70L*x0))];
                auto tmp1 = out_ptr0[static_cast<long>(x1)];
                auto tmp3 = out_ptr1[static_cast<long>(x1)];
                auto tmp10 = in_ptr1[static_cast<long>(x1)];
                auto tmp12 = in_ptr2[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(8.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr2[static_cast<long>(x1 + (70L*x0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused_hardtanh_mul_sigmoid_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(840L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (840L*x1) + (41160L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (840L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = at::vec::maximum(tmp3, tmp5);
                        auto tmp7 = static_cast<float>(6.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = at::vec::minimum(tmp6, tmp8);
                        tmp9.store(out_ptr0 + static_cast<long>(x2 + (840L*x1) + (41160L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_cat_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(144L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (151L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(144L); x0<static_cast<long>(151L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                Welford<float> tmp_acc0 = Welford<float>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (151L*x1))];
                    tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(144L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(144L); x0<static_cast<long>(151L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
            auto tmp5 = 1 / std::sqrt(tmp4);
            out_ptr2[static_cast<long>(x0)] = tmp5;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(151L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (151L*x0))];
                auto tmp1 = out_ptr0[static_cast<long>(x1)];
                auto tmp3 = out_ptr1[static_cast<long>(x1)];
                auto tmp10 = in_ptr1[static_cast<long>(x1)];
                auto tmp12 = in_ptr2[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(392.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                auto tmp14 = c10::convert<long>(x1);
                auto tmp15 = static_cast<long>(0);
                auto tmp16 = tmp14 >= tmp15;
                auto tmp17 = static_cast<long>(140);
                auto tmp18 = tmp14 < tmp17;
                auto tmp19 = [&]
                {
                    auto tmp20 = in_ptr3[static_cast<long>(x1 + (140L*x0))];
                    auto tmp21 = decltype(tmp13)(tmp13 + tmp20);
                    return tmp21;
                }
                ;
                auto tmp22 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                auto tmp23 = tmp14 >= tmp17;
                auto tmp24 = static_cast<long>(151);
                auto tmp25 = tmp14 < tmp24;
                auto tmp26 = [&]
                {
                    return tmp13;
                }
                ;
                auto tmp27 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                auto tmp28 = tmp18 ? tmp22 : tmp27;
                in_out_ptr0[static_cast<long>(x1 + (151L*x0))] = tmp28;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_silu_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(904L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (906L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(904L); x0<static_cast<long>(906L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (906L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(904L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(904L); x0<static_cast<long>(906L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(904L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (906L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(392.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (906L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(904L); x1<static_cast<long>(906L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (906L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(392.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    out_ptr3[static_cast<long>(x1 + (906L*x0))] = tmp13;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(355152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_mean_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(904L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (906L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(904L); x0<static_cast<long>(906L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (906L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(904L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(904L); x0<static_cast<long>(906L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(904L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (906L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(392.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (906L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(904L); x1<static_cast<long>(906L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (906L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(392.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    out_ptr3[static_cast<long>(x1 + (906L*x0))] = tmp13;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(904L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1 + (906L*x2) + (44394L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x1 + (906L*x0)));
                    }
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(904L); x1<static_cast<long>(906L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr3[static_cast<long>(x1 + (906L*x2) + (44394L*x0))];
                            tmp_acc0 = tmp_acc0 + tmp0;
                        }
                        out_ptr4[static_cast<long>(x1 + (906L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(7248L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(49.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (75L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(72L); x0<static_cast<long>(75L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                Welford<float> tmp_acc0 = Welford<float>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (75L*x1))];
                    tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (75L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(8.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp11 = tmp2 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                tmp16.store(out_ptr2 + static_cast<long>(x1 + (75L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(72L); x1<static_cast<long>(75L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (75L*x0))];
                auto tmp1 = out_ptr0[static_cast<long>(x1)];
                auto tmp3 = out_ptr1[static_cast<long>(x1)];
                auto tmp10 = in_ptr1[static_cast<long>(x1)];
                auto tmp12 = in_ptr2[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(8.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr2[static_cast<long>(x1 + (75L*x0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused_hardtanh_mul_sigmoid_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(904L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (906L*x1) + (44394L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (906L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = at::vec::maximum(tmp3, tmp5);
                        auto tmp7 = static_cast<float>(6.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = at::vec::minimum(tmp6, tmp8);
                        tmp9.store(out_ptr0 + static_cast<long>(x2 + (906L*x1) + (44394L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(904L); x2<static_cast<long>(906L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (906L*x1) + (44394L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (906L*x0))];
                        auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                        auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = max_propagate_nan(tmp3, tmp4);
                        auto tmp6 = static_cast<float>(6.0);
                        auto tmp7 = min_propagate_nan(tmp5, tmp6);
                        out_ptr0[static_cast<long>(x2 + (906L*x1) + (44394L*x0))] = tmp7;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_cat_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (162L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(160L); x0<static_cast<long>(162L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                Welford<float> tmp_acc0 = Welford<float>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (162L*x1))];
                    tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(160L); x0<static_cast<long>(162L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
            auto tmp5 = 1 / std::sqrt(tmp4);
            out_ptr2[static_cast<long>(x0)] = tmp5;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(162L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (162L*x0))];
                auto tmp1 = out_ptr0[static_cast<long>(x1)];
                auto tmp3 = out_ptr1[static_cast<long>(x1)];
                auto tmp10 = in_ptr1[static_cast<long>(x1)];
                auto tmp12 = in_ptr2[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(392.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                auto tmp14 = c10::convert<long>(x1);
                auto tmp15 = static_cast<long>(0);
                auto tmp16 = tmp14 >= tmp15;
                auto tmp17 = static_cast<long>(151);
                auto tmp18 = tmp14 < tmp17;
                auto tmp19 = [&]
                {
                    auto tmp20 = in_ptr3[static_cast<long>(x1 + (151L*x0))];
                    auto tmp21 = decltype(tmp13)(tmp13 + tmp20);
                    return tmp21;
                }
                ;
                auto tmp22 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                auto tmp23 = tmp14 >= tmp17;
                auto tmp24 = static_cast<long>(162);
                auto tmp25 = tmp14 < tmp24;
                auto tmp26 = [&]
                {
                    return tmp13;
                }
                ;
                auto tmp27 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                auto tmp28 = tmp18 ? tmp22 : tmp27;
                in_out_ptr0[static_cast<long>(x1 + (162L*x0))] = tmp28;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(968L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (972L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(968L); x0<static_cast<long>(972L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (972L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(968L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(968L); x0<static_cast<long>(972L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(968L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (972L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(392.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (972L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(968L); x1<static_cast<long>(972L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (972L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(392.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    out_ptr3[static_cast<long>(x1 + (972L*x0))] = tmp13;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(381024L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                auto tmp3 = static_cast<float>(1.0);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp4 - tmp1;
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6 + tmp4;
                auto tmp8 = tmp1 * tmp7;
                tmp2.store(out_ptr4 + static_cast<long>(x0));
                tmp8.store(out_ptr5 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_mean_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(968L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (972L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(968L); x0<static_cast<long>(972L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (972L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(968L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(968L); x0<static_cast<long>(972L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(968L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (972L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(392.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (972L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(968L); x1<static_cast<long>(972L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (972L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(392.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    out_ptr3[static_cast<long>(x1 + (972L*x0))] = tmp13;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(968L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1 + (972L*x2) + (47628L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x1 + (972L*x0)));
                    }
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(968L); x1<static_cast<long>(972L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr3[static_cast<long>(x1 + (972L*x2) + (47628L*x0))];
                            tmp_acc0 = tmp_acc0 + tmp0;
                        }
                        out_ptr4[static_cast<long>(x1 + (972L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(7776L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(49.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (81L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(80L); x0<static_cast<long>(81L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                Welford<float> tmp_acc0 = Welford<float>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (81L*x1))];
                    tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (81L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(8.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp11 = tmp2 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                tmp16.store(out_ptr2 + static_cast<long>(x1 + (81L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(80L); x1<static_cast<long>(81L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (81L*x0))];
                auto tmp1 = out_ptr0[static_cast<long>(x1)];
                auto tmp3 = out_ptr1[static_cast<long>(x1)];
                auto tmp10 = in_ptr1[static_cast<long>(x1)];
                auto tmp12 = in_ptr2[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(8.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr2[static_cast<long>(x1 + (81L*x0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused_hardtanh_mul_sigmoid_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(968L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (972L*x1) + (47628L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (972L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = at::vec::maximum(tmp3, tmp5);
                        auto tmp7 = static_cast<float>(6.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = at::vec::minimum(tmp6, tmp8);
                        tmp9.store(out_ptr0 + static_cast<long>(x2 + (972L*x1) + (47628L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(968L); x2<static_cast<long>(972L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (972L*x1) + (47628L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (972L*x0))];
                        auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                        auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = max_propagate_nan(tmp3, tmp4);
                        auto tmp6 = static_cast<float>(6.0);
                        auto tmp7 = min_propagate_nan(tmp5, tmp6);
                        out_ptr0[static_cast<long>(x2 + (972L*x1) + (47628L*x0))] = tmp7;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_cat_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(168L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (174L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(168L); x0<static_cast<long>(174L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                Welford<float> tmp_acc0 = Welford<float>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (174L*x1))];
                    tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(168L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(168L); x0<static_cast<long>(174L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
            auto tmp5 = 1 / std::sqrt(tmp4);
            out_ptr2[static_cast<long>(x0)] = tmp5;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(174L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (174L*x0))];
                auto tmp1 = out_ptr0[static_cast<long>(x1)];
                auto tmp3 = out_ptr1[static_cast<long>(x1)];
                auto tmp10 = in_ptr1[static_cast<long>(x1)];
                auto tmp12 = in_ptr2[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(392.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                auto tmp14 = c10::convert<long>(x1);
                auto tmp15 = static_cast<long>(0);
                auto tmp16 = tmp14 >= tmp15;
                auto tmp17 = static_cast<long>(162);
                auto tmp18 = tmp14 < tmp17;
                auto tmp19 = [&]
                {
                    auto tmp20 = in_ptr3[static_cast<long>(x1 + (162L*x0))];
                    auto tmp21 = decltype(tmp13)(tmp13 + tmp20);
                    return tmp21;
                }
                ;
                auto tmp22 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                auto tmp23 = tmp14 >= tmp17;
                auto tmp24 = static_cast<long>(174);
                auto tmp25 = tmp14 < tmp24;
                auto tmp26 = [&]
                {
                    return tmp13;
                }
                ;
                auto tmp27 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                auto tmp28 = tmp18 ? tmp22 : tmp27;
                in_out_ptr0[static_cast<long>(x1 + (174L*x0))] = tmp28;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1040L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1044L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(1040L); x0<static_cast<long>(1044L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (1044L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1040L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(1040L); x0<static_cast<long>(1044L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1040L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1044L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(392.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (1044L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(1040L); x1<static_cast<long>(1044L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (1044L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(392.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    out_ptr3[static_cast<long>(x1 + (1044L*x0))] = tmp13;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(409248L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                auto tmp3 = static_cast<float>(1.0);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp4 - tmp1;
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6 + tmp4;
                auto tmp8 = tmp1 * tmp7;
                tmp2.store(out_ptr4 + static_cast<long>(x0));
                tmp8.store(out_ptr5 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_mean_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1040L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1044L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(1040L); x0<static_cast<long>(1044L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (1044L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1040L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(1040L); x0<static_cast<long>(1044L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1040L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1044L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(392.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (1044L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(1040L); x1<static_cast<long>(1044L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (1044L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(392.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    out_ptr3[static_cast<long>(x1 + (1044L*x0))] = tmp13;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1040L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1 + (1044L*x2) + (51156L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x1 + (1044L*x0)));
                    }
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(1040L); x1<static_cast<long>(1044L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr3[static_cast<long>(x1 + (1044L*x2) + (51156L*x0))];
                            tmp_acc0 = tmp_acc0 + tmp0;
                        }
                        out_ptr4[static_cast<long>(x1 + (1044L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8352L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(49.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (87L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(80L); x0<static_cast<long>(87L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                Welford<float> tmp_acc0 = Welford<float>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (87L*x1))];
                    tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (87L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(8.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp11 = tmp2 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                tmp16.store(out_ptr2 + static_cast<long>(x1 + (87L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(80L); x1<static_cast<long>(87L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (87L*x0))];
                auto tmp1 = out_ptr0[static_cast<long>(x1)];
                auto tmp3 = out_ptr1[static_cast<long>(x1)];
                auto tmp10 = in_ptr1[static_cast<long>(x1)];
                auto tmp12 = in_ptr2[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(8.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                auto tmp14 = tmp13 * (tmp13>0);
                out_ptr2[static_cast<long>(x1 + (87L*x0))] = tmp14;
            }
        }
    }
}
''')


cpp_fused_hardtanh_mul_sigmoid_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1040L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1044L*x1) + (51156L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1044L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = at::vec::maximum(tmp3, tmp5);
                        auto tmp7 = static_cast<float>(6.0);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = at::vec::minimum(tmp6, tmp8);
                        tmp9.store(out_ptr0 + static_cast<long>(x2 + (1044L*x1) + (51156L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(1040L); x2<static_cast<long>(1044L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (1044L*x1) + (51156L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (1044L*x0))];
                        auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                        auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = max_propagate_nan(tmp3, tmp4);
                        auto tmp6 = static_cast<float>(6.0);
                        auto tmp7 = min_propagate_nan(tmp5, tmp6);
                        out_ptr0[static_cast<long>(x2 + (1044L*x1) + (51156L*x0))] = tmp7;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_cat_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (185L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(184L); x0<static_cast<long>(185L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                Welford<float> tmp_acc0 = Welford<float>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (185L*x1))];
                    tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(184L); x0<static_cast<long>(185L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1e-05);
            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
            auto tmp5 = 1 / std::sqrt(tmp4);
            out_ptr2[static_cast<long>(x0)] = tmp5;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(185L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (185L*x0))];
                auto tmp1 = out_ptr0[static_cast<long>(x1)];
                auto tmp3 = out_ptr1[static_cast<long>(x1)];
                auto tmp10 = in_ptr1[static_cast<long>(x1)];
                auto tmp12 = in_ptr2[static_cast<long>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                auto tmp4 = static_cast<float>(392.0);
                auto tmp5 = tmp3 / tmp4;
                auto tmp6 = static_cast<float>(1e-05);
                auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                auto tmp8 = 1 / std::sqrt(tmp7);
                auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                auto tmp14 = c10::convert<long>(x1);
                auto tmp15 = static_cast<long>(0);
                auto tmp16 = tmp14 >= tmp15;
                auto tmp17 = static_cast<long>(174);
                auto tmp18 = tmp14 < tmp17;
                auto tmp19 = [&]
                {
                    auto tmp20 = in_ptr3[static_cast<long>(x1 + (174L*x0))];
                    auto tmp21 = decltype(tmp13)(tmp13 + tmp20);
                    return tmp21;
                }
                ;
                auto tmp22 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                auto tmp23 = tmp14 >= tmp17;
                auto tmp24 = static_cast<long>(185);
                auto tmp25 = tmp14 < tmp24;
                auto tmp26 = [&]
                {
                    return tmp13;
                }
                ;
                auto tmp27 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                auto tmp28 = tmp18 ? tmp22 : tmp27;
                in_out_ptr0[static_cast<long>(x1 + (185L*x0))] = tmp28;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_mean_silu_view_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1280L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1280L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1280L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1280L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1280L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(392.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (1280L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1280L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1 + (1280L*x2) + (62720L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x1 + (1280L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(10240L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(49.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_fill_hardtanh_backward_mul_sigmoid_sub_76 = async_compile.cpp('''
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
                       float* in_out_ptr72,
                       float* in_out_ptr73,
                       float* in_out_ptr74,
                       float* in_out_ptr75,
                       float* in_out_ptr76,
                       float* in_out_ptr77,
                       float* in_out_ptr78,
                       float* in_out_ptr79,
                       float* in_out_ptr80,
                       float* in_out_ptr81,
                       float* in_out_ptr82,
                       float* in_out_ptr83,
                       float* in_out_ptr84,
                       float* in_out_ptr85,
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
                       const float* in_ptr199,
                       const float* in_ptr200,
                       const long* in_ptr201,
                       const float* in_ptr202,
                       const float* in_ptr203,
                       const long* in_ptr204,
                       const float* in_ptr205,
                       const float* in_ptr206,
                       const long* in_ptr207,
                       const float* in_ptr208,
                       const float* in_ptr209,
                       const long* in_ptr210,
                       const float* in_ptr211,
                       const float* in_ptr212,
                       const long* in_ptr213,
                       const float* in_ptr214,
                       const float* in_ptr215,
                       const long* in_ptr216,
                       const float* in_ptr217,
                       const float* in_ptr218,
                       const long* in_ptr219,
                       const float* in_ptr220,
                       const float* in_ptr221,
                       const long* in_ptr222,
                       const float* in_ptr223,
                       const float* in_ptr224,
                       const long* in_ptr225,
                       const float* in_ptr226,
                       const float* in_ptr227,
                       const long* in_ptr228,
                       const float* in_ptr229,
                       const float* in_ptr230,
                       const long* in_ptr231,
                       bool* out_ptr0,
                       bool* out_ptr1,
                       bool* out_ptr2,
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
                       float* out_ptr248,
                       float* out_ptr249,
                       long* out_ptr251,
                       float* out_ptr252,
                       float* out_ptr253,
                       long* out_ptr255,
                       float* out_ptr256,
                       float* out_ptr257,
                       long* out_ptr259,
                       float* out_ptr260,
                       float* out_ptr261,
                       long* out_ptr263,
                       float* out_ptr264,
                       float* out_ptr265,
                       long* out_ptr267,
                       float* out_ptr268,
                       float* out_ptr269,
                       long* out_ptr271,
                       float* out_ptr272,
                       float* out_ptr273,
                       long* out_ptr275,
                       float* out_ptr276,
                       float* out_ptr277,
                       long* out_ptr279,
                       float* out_ptr280,
                       float* out_ptr281,
                       long* out_ptr283,
                       float* out_ptr284,
                       float* out_ptr285,
                       long* out_ptr287,
                       float* out_ptr288,
                       float* out_ptr289,
                       long* out_ptr291)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(501760L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(355152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr1 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(329280L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr2 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1204224L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr3 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr3 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1100736L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr4 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr4 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(997248L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr5 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr5 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(893760L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr6 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr6 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(790272L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr7 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr7 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(677376L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr8 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr8 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2295552L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr9 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr9 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1881600L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr10 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr10 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(5720064L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr11 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr11 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4064256L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp3 = static_cast<float>(6.0);
                auto tmp4 = tmp0 >= tmp3;
                auto tmp5 = decltype(tmp2)(tmp2 | tmp4);
                out_ptr0[static_cast<long>(x0)] = tmp5;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4064256L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr12 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr12 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr1[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp3 = static_cast<float>(6.0);
                auto tmp4 = tmp0 >= tmp3;
                auto tmp5 = decltype(tmp2)(tmp2 | tmp4);
                out_ptr1[static_cast<long>(x0)] = tmp5;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9633792L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr13 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr13 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr2[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp3 = static_cast<float>(6.0);
                auto tmp4 = tmp0 >= tmp3;
                auto tmp5 = decltype(tmp2)(tmp2 | tmp4);
                out_ptr2[static_cast<long>(x0)] = tmp5;
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr14 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = static_cast<float>(1.0);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp3 - tmp1;
                auto tmp5 = tmp0 * tmp4;
                auto tmp6 = tmp5 + tmp3;
                auto tmp7 = tmp1 * tmp6;
                tmp7.store(in_out_ptr14 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr3[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr4[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr15 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr7 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(100352.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.00000996502277);
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
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr7[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr9[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr16 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr12 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(100352.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.00000996502277);
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
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr11[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr14[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr17 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr17 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(100352.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.00000996502277);
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
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr15[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr19[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr18 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr22 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(100352.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.00000996502277);
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
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr19[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr24[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr19 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr27 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0000398612827361);
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
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr23[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr29[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(24L); x0<static_cast<long>(27L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr24[static_cast<long>(x0)];
                    auto tmp3 = in_ptr25[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr31[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr20 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr32 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0000398612827361);
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(24L); x0<static_cast<long>(27L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr20[static_cast<long>(x0)];
                    auto tmp7 = in_ptr26[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0000398612827361);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr32[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr27[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr34[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(160L); x0<static_cast<long>(162L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr28[static_cast<long>(x0)];
                    auto tmp3 = in_ptr29[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr36[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr21 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr37 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0000398612827361);
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(160L); x0<static_cast<long>(162L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr21[static_cast<long>(x0)];
                    auto tmp7 = in_ptr30[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0000398612827361);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr37[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr31[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr39[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(160L); x0<static_cast<long>(162L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr32[static_cast<long>(x0)];
                    auto tmp3 = in_ptr33[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr41[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr22 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr42 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0000398612827361);
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(160L); x0<static_cast<long>(162L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr22[static_cast<long>(x0)];
                    auto tmp7 = in_ptr34[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0000398612827361);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr42[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr35[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr44[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(32L); x0<static_cast<long>(38L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr36[static_cast<long>(x0)];
                    auto tmp3 = in_ptr37[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr46[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr23 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr47 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0000398612827361);
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(32L); x0<static_cast<long>(38L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr23[static_cast<long>(x0)];
                    auto tmp7 = in_ptr38[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0000398612827361);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr47[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr39[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr49[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(224L); x0<static_cast<long>(228L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr40[static_cast<long>(x0)];
                    auto tmp3 = in_ptr41[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr51[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr24 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr52 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0000398612827361);
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(224L); x0<static_cast<long>(228L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr24[static_cast<long>(x0)];
                    auto tmp7 = in_ptr42[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0000398612827361);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr52[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr43[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr54[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(224L); x0<static_cast<long>(228L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr44[static_cast<long>(x0)];
                    auto tmp3 = in_ptr45[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr56[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr25 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr57 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0001594642002871);
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(224L); x0<static_cast<long>(228L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr25[static_cast<long>(x0)];
                    auto tmp7 = in_ptr46[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0001594642002871);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr57[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr47[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr59[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(48L); x0<static_cast<long>(50L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr48[static_cast<long>(x0)];
                    auto tmp3 = in_ptr49[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr61[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr26 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr62 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0001594642002871);
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(48L); x0<static_cast<long>(50L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr26[static_cast<long>(x0)];
                    auto tmp7 = in_ptr50[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0001594642002871);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr62[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr51[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr64[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(296L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(296L); x0<static_cast<long>(300L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr52[static_cast<long>(x0)];
                    auto tmp3 = in_ptr53[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr66[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(296L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr27 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr67 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0001594642002871);
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(296L); x0<static_cast<long>(300L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr27[static_cast<long>(x0)];
                    auto tmp7 = in_ptr54[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0001594642002871);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr67[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr55[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr69[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(296L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(296L); x0<static_cast<long>(300L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr56[static_cast<long>(x0)];
                    auto tmp3 = in_ptr57[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr71[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(296L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr28 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr72 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0001594642002871);
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(296L); x0<static_cast<long>(300L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr28[static_cast<long>(x0)];
                    auto tmp7 = in_ptr58[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0001594642002871);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr72[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr59[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr74[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(56L); x0<static_cast<long>(61L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr60[static_cast<long>(x0)];
                    auto tmp3 = in_ptr61[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr76[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr29 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr77 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0001594642002871);
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(56L); x0<static_cast<long>(61L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr29[static_cast<long>(x0)];
                    auto tmp7 = in_ptr62[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0001594642002871);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr77[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr63[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr79[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(360L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(360L); x0<static_cast<long>(366L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr64[static_cast<long>(x0)];
                    auto tmp3 = in_ptr65[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr81[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(360L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr30 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr82 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0001594642002871);
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(360L); x0<static_cast<long>(366L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr30[static_cast<long>(x0)];
                    auto tmp7 = in_ptr66[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0001594642002871);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr82[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr67[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr84[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(360L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(360L); x0<static_cast<long>(366L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr68[static_cast<long>(x0)];
                    auto tmp3 = in_ptr69[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr86[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(360L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr31 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr87 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(360L); x0<static_cast<long>(366L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr31[static_cast<long>(x0)];
                    auto tmp7 = in_ptr70[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0006381620931717);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr87[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr71[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr89[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr32 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr92 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
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
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr75[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr94[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(432L); x0+=static_cast<long>(8L))
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(432L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr33 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr97 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
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
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr79[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr99[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(432L); x0+=static_cast<long>(8L))
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(432L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr34 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr102 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
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
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr83[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr104[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(80L); x0<static_cast<long>(84L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr84[static_cast<long>(x0)];
                    auto tmp3 = in_ptr85[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr106[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr35 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr107 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(80L); x0<static_cast<long>(84L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr35[static_cast<long>(x0)];
                    auto tmp7 = in_ptr86[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0006381620931717);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr107[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr87[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr109[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(504L); x0+=static_cast<long>(8L))
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(504L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr36 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr112 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
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
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr91[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr114[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(504L); x0+=static_cast<long>(8L))
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(504L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr37 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr117 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
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
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr95[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr119[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(88L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(88L); x0<static_cast<long>(95L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr96[static_cast<long>(x0)];
                    auto tmp3 = in_ptr97[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr121[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(88L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr38 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr122 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(88L); x0<static_cast<long>(95L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr38[static_cast<long>(x0)];
                    auto tmp7 = in_ptr98[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0006381620931717);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr122[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr99[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr124[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(568L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(568L); x0<static_cast<long>(570L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr100[static_cast<long>(x0)];
                    auto tmp3 = in_ptr101[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr126[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr39 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr127 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(568L); x0<static_cast<long>(570L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr39[static_cast<long>(x0)];
                    auto tmp7 = in_ptr102[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0006381620931717);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr127[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr103[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr129[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(568L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(568L); x0<static_cast<long>(570L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr104[static_cast<long>(x0)];
                    auto tmp3 = in_ptr105[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr131[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr40 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr132 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(568L); x0<static_cast<long>(570L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr40[static_cast<long>(x0)];
                    auto tmp7 = in_ptr106[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0006381620931717);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr132[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr107[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr134[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(104L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(104L); x0<static_cast<long>(106L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr108[static_cast<long>(x0)];
                    auto tmp3 = in_ptr109[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr136[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(104L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr41 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr137 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(104L); x0<static_cast<long>(106L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr41[static_cast<long>(x0)];
                    auto tmp7 = in_ptr110[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0006381620931717);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr137[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr111[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr139[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(632L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(632L); x0<static_cast<long>(636L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr112[static_cast<long>(x0)];
                    auto tmp3 = in_ptr113[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr141[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(632L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr42 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr142 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(632L); x0<static_cast<long>(636L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr42[static_cast<long>(x0)];
                    auto tmp7 = in_ptr114[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0006381620931717);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr142[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr115[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr144[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(632L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(632L); x0<static_cast<long>(636L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr116[static_cast<long>(x0)];
                    auto tmp3 = in_ptr117[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr146[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(632L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr43 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr147 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(632L); x0<static_cast<long>(636L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr43[static_cast<long>(x0)];
                    auto tmp7 = in_ptr118[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0006381620931717);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr147[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr119[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr149[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(112L); x0<static_cast<long>(117L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr120[static_cast<long>(x0)];
                    auto tmp3 = in_ptr121[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr151[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr44 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr152 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(112L); x0<static_cast<long>(117L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr44[static_cast<long>(x0)];
                    auto tmp7 = in_ptr122[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0006381620931717);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr152[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr123[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr154[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(696L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(696L); x0<static_cast<long>(702L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr124[static_cast<long>(x0)];
                    auto tmp3 = in_ptr125[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr156[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(696L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr45 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr157 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(696L); x0<static_cast<long>(702L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr45[static_cast<long>(x0)];
                    auto tmp7 = in_ptr126[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0006381620931717);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr157[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr127[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr159[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(696L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(696L); x0<static_cast<long>(702L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr128[static_cast<long>(x0)];
                    auto tmp3 = in_ptr129[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr161[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(696L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr46 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr162 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(696L); x0<static_cast<long>(702L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr46[static_cast<long>(x0)];
                    auto tmp7 = in_ptr130[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0006381620931717);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr162[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr131[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr164[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr47 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr167 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
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
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr135[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr169[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr48 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr172 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
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
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr139[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr174[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr49 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr177 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
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
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr143[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr179[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(136L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(136L); x0<static_cast<long>(140L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr144[static_cast<long>(x0)];
                    auto tmp3 = in_ptr145[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr181[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(136L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr50 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr182 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(136L); x0<static_cast<long>(140L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr50[static_cast<long>(x0)];
                    auto tmp7 = in_ptr146[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0025575447570332);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr182[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr147[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr184[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(840L); x0+=static_cast<long>(8L))
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(840L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr51 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr187 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
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
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr151[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr189[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(840L); x0+=static_cast<long>(8L))
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(840L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr52 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr192 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
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
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr155[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr194[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(144L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(144L); x0<static_cast<long>(151L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr156[static_cast<long>(x0)];
                    auto tmp3 = in_ptr157[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr196[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(144L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr53 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr197 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(144L); x0<static_cast<long>(151L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr53[static_cast<long>(x0)];
                    auto tmp7 = in_ptr158[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0025575447570332);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr197[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr159[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr199[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(904L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(904L); x0<static_cast<long>(906L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr160[static_cast<long>(x0)];
                    auto tmp3 = in_ptr161[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr201[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(904L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr54 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr202 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(904L); x0<static_cast<long>(906L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr54[static_cast<long>(x0)];
                    auto tmp7 = in_ptr162[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0025575447570332);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr202[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr163[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr204[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(904L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(904L); x0<static_cast<long>(906L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr164[static_cast<long>(x0)];
                    auto tmp3 = in_ptr165[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr206[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(904L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr55 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr207 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(904L); x0<static_cast<long>(906L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr55[static_cast<long>(x0)];
                    auto tmp7 = in_ptr166[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0025575447570332);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr207[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr167[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr209[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(160L); x0<static_cast<long>(162L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr168[static_cast<long>(x0)];
                    auto tmp3 = in_ptr169[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr211[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr56 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr212 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(160L); x0<static_cast<long>(162L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr56[static_cast<long>(x0)];
                    auto tmp7 = in_ptr170[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0025575447570332);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr212[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr171[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr214[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(968L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(968L); x0<static_cast<long>(972L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr172[static_cast<long>(x0)];
                    auto tmp3 = in_ptr173[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr216[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(968L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr57 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr217 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(968L); x0<static_cast<long>(972L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr57[static_cast<long>(x0)];
                    auto tmp7 = in_ptr174[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0025575447570332);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr217[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr175[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr219[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(968L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(968L); x0<static_cast<long>(972L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr176[static_cast<long>(x0)];
                    auto tmp3 = in_ptr177[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr221[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(968L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr58 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr222 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(968L); x0<static_cast<long>(972L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr58[static_cast<long>(x0)];
                    auto tmp7 = in_ptr178[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0025575447570332);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr222[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr179[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr224[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(168L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(168L); x0<static_cast<long>(174L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr180[static_cast<long>(x0)];
                    auto tmp3 = in_ptr181[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr226[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(168L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr59 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr227 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(168L); x0<static_cast<long>(174L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr59[static_cast<long>(x0)];
                    auto tmp7 = in_ptr182[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0025575447570332);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr227[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr183[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr229[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1040L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(1040L); x0<static_cast<long>(1044L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr184[static_cast<long>(x0)];
                    auto tmp3 = in_ptr185[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr231[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1040L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr60 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr232 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(1040L); x0<static_cast<long>(1044L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr60[static_cast<long>(x0)];
                    auto tmp7 = in_ptr186[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0025575447570332);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr232[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr187[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr234[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1040L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(1040L); x0<static_cast<long>(1044L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr188[static_cast<long>(x0)];
                    auto tmp3 = in_ptr189[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr236[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1040L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr61 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr237 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(1040L); x0<static_cast<long>(1044L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr61[static_cast<long>(x0)];
                    auto tmp7 = in_ptr190[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0025575447570332);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr237[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr191[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr239[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(184L); x0<static_cast<long>(185L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr192[static_cast<long>(x0)];
                    auto tmp3 = in_ptr193[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr241[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr62 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr242 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(184L); x0<static_cast<long>(185L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr62[static_cast<long>(x0)];
                    auto tmp7 = in_ptr194[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0025575447570332);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr242[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr195[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr244[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1280L); x0+=static_cast<long>(8L))
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1280L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr63 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr247 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr64 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr248 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr248 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(16L); x0<static_cast<long>(19L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr64[static_cast<long>(x0)];
                    auto tmp3 = in_ptr199[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr248[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr65 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr249 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.1428571428571428);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr249 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(16L); x0<static_cast<long>(19L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr65[static_cast<long>(x0)];
                    auto tmp7 = in_ptr200[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(8.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.1428571428571428);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr249[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr201[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr251[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr66 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr252 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr252 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(24L); x0<static_cast<long>(25L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr66[static_cast<long>(x0)];
                    auto tmp3 = in_ptr202[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr252[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr67 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr253 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.1428571428571428);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr253 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(24L); x0<static_cast<long>(25L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr67[static_cast<long>(x0)];
                    auto tmp7 = in_ptr203[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(8.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.1428571428571428);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr253[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr204[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr255[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr68 + static_cast<long>(x0));
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(24L); x0<static_cast<long>(30L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr68[static_cast<long>(x0)];
                    auto tmp3 = in_ptr205[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr256[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr69 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr257 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.1428571428571428);
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(24L); x0<static_cast<long>(30L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr69[static_cast<long>(x0)];
                    auto tmp7 = in_ptr206[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(8.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.1428571428571428);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr257[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr207[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr259[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr70 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr260 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr260 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(32L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr70[static_cast<long>(x0)];
                    auto tmp3 = in_ptr208[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr260[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr71 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr261 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.1428571428571428);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr261 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(32L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr71[static_cast<long>(x0)];
                    auto tmp7 = in_ptr209[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(8.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.1428571428571428);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr261[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr210[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr263[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr72 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr264 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr264 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(40L); x0<static_cast<long>(42L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr72[static_cast<long>(x0)];
                    auto tmp3 = in_ptr211[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr264[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr73 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr265 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.1428571428571428);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr265 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(40L); x0<static_cast<long>(42L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr73[static_cast<long>(x0)];
                    auto tmp7 = in_ptr212[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(8.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.1428571428571428);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr265[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr213[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr267[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr74 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr268 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr268 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(40L); x0<static_cast<long>(47L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr74[static_cast<long>(x0)];
                    auto tmp3 = in_ptr214[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr268[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr75 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr269 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.1428571428571428);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr269 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(40L); x0<static_cast<long>(47L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr75[static_cast<long>(x0)];
                    auto tmp7 = in_ptr215[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(8.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.1428571428571428);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr269[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr216[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr271[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr76 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr272 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr272 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(48L); x0<static_cast<long>(53L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr76[static_cast<long>(x0)];
                    auto tmp3 = in_ptr217[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr272[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr77 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr273 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.1428571428571428);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr273 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(48L); x0<static_cast<long>(53L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr77[static_cast<long>(x0)];
                    auto tmp7 = in_ptr218[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(8.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.1428571428571428);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr273[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr219[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr275[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr78 + static_cast<long>(x0));
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(56L); x0<static_cast<long>(58L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr78[static_cast<long>(x0)];
                    auto tmp3 = in_ptr220[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr276[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr79 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr277 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.1428571428571428);
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
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(56L); x0<static_cast<long>(58L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr79[static_cast<long>(x0)];
                    auto tmp7 = in_ptr221[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(8.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.1428571428571428);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr277[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr222[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr279[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr80 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr280 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr280 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr81 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr281 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.1428571428571428);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr281 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr225[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr283[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr82 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr284 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr284 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(64L); x0<static_cast<long>(70L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr82[static_cast<long>(x0)];
                    auto tmp3 = in_ptr226[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr284[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr83 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr285 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.1428571428571428);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr285 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(64L); x0<static_cast<long>(70L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr83[static_cast<long>(x0)];
                    auto tmp7 = in_ptr227[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(8.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.1428571428571428);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr285[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr228[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr287[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr84 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr288 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr288 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(72L); x0<static_cast<long>(75L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr84[static_cast<long>(x0)];
                    auto tmp3 = in_ptr229[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr288[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr85 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr289 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(8.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.1428571428571428);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr289 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(72L); x0<static_cast<long>(75L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr85[static_cast<long>(x0)];
                    auto tmp7 = in_ptr230[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(8.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.1428571428571428);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr289[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr231[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr291[static_cast<long>(0L)] = tmp2;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       float* in_out_ptr3,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const long* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const long* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       long* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       long* out_ptr7)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr0 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(80L); x0<static_cast<long>(81L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_out_ptr0[static_cast<long>(x0)];
            auto tmp3 = in_ptr0[static_cast<long>(x0)];
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
            auto tmp4 = static_cast<float>(0.9);
            auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
            auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
            out_ptr0[static_cast<long>(x0)] = tmp6;
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.1428571428571428);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr1 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(80L); x0<static_cast<long>(81L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_out_ptr1[static_cast<long>(x0)];
            auto tmp7 = in_ptr1[static_cast<long>(x0)];
            auto tmp1 = static_cast<float>(8.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.1428571428571428);
            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
            out_ptr1[static_cast<long>(x0)] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr2[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr3[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr4 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(80L); x0<static_cast<long>(87L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_out_ptr2[static_cast<long>(x0)];
            auto tmp3 = in_ptr3[static_cast<long>(x0)];
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
            auto tmp4 = static_cast<float>(0.9);
            auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
            auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
            out_ptr4[static_cast<long>(x0)] = tmp6;
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr3 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.1428571428571428);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr5 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(80L); x0<static_cast<long>(87L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_out_ptr3[static_cast<long>(x0)];
            auto tmp7 = in_ptr4[static_cast<long>(x0)];
            auto tmp1 = static_cast<float>(8.0);
            auto tmp2 = tmp0 / tmp1;
            auto tmp3 = static_cast<float>(1.1428571428571428);
            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
            auto tmp5 = static_cast<float>(0.1);
            auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
            auto tmp8 = static_cast<float>(0.9);
            auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
            auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
            out_ptr5[static_cast<long>(x0)] = tmp10;
        }
    }
    {
        auto tmp0 = in_ptr5[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr7[static_cast<long>(0L)] = tmp2;
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414 = args
    args.clear()
    assert_size_stride(primals_1, (32, ), (1, ))
    assert_size_stride(primals_2, (32, ), (1, ))
    assert_size_stride(primals_3, (32, ), (1, ))
    assert_size_stride(primals_4, (32, ), (1, ))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_6, (16, ), (1, ))
    assert_size_stride(primals_7, (96, ), (1, ))
    assert_size_stride(primals_8, (96, ), (1, ))
    assert_size_stride(primals_9, (96, ), (1, ))
    assert_size_stride(primals_10, (96, ), (1, ))
    assert_size_stride(primals_11, (27, ), (1, ))
    assert_size_stride(primals_12, (27, ), (1, ))
    assert_size_stride(primals_13, (162, ), (1, ))
    assert_size_stride(primals_14, (162, ), (1, ))
    assert_size_stride(primals_15, (162, ), (1, ))
    assert_size_stride(primals_16, (162, ), (1, ))
    assert_size_stride(primals_17, (38, ), (1, ))
    assert_size_stride(primals_18, (38, ), (1, ))
    assert_size_stride(primals_19, (228, ), (1, ))
    assert_size_stride(primals_20, (228, ), (1, ))
    assert_size_stride(primals_21, (228, ), (1, ))
    assert_size_stride(primals_22, (228, ), (1, ))
    assert_size_stride(primals_23, (50, ), (1, ))
    assert_size_stride(primals_24, (50, ), (1, ))
    assert_size_stride(primals_25, (300, ), (1, ))
    assert_size_stride(primals_26, (300, ), (1, ))
    assert_size_stride(primals_27, (300, ), (1, ))
    assert_size_stride(primals_28, (300, ), (1, ))
    assert_size_stride(primals_29, (61, ), (1, ))
    assert_size_stride(primals_30, (61, ), (1, ))
    assert_size_stride(primals_31, (366, ), (1, ))
    assert_size_stride(primals_32, (366, ), (1, ))
    assert_size_stride(primals_33, (366, ), (1, ))
    assert_size_stride(primals_34, (366, ), (1, ))
    assert_size_stride(primals_35, (72, ), (1, ))
    assert_size_stride(primals_36, (72, ), (1, ))
    assert_size_stride(primals_37, (432, ), (1, ))
    assert_size_stride(primals_38, (432, ), (1, ))
    assert_size_stride(primals_39, (432, ), (1, ))
    assert_size_stride(primals_40, (432, ), (1, ))
    assert_size_stride(primals_41, (84, ), (1, ))
    assert_size_stride(primals_42, (84, ), (1, ))
    assert_size_stride(primals_43, (504, ), (1, ))
    assert_size_stride(primals_44, (504, ), (1, ))
    assert_size_stride(primals_45, (504, ), (1, ))
    assert_size_stride(primals_46, (504, ), (1, ))
    assert_size_stride(primals_47, (95, ), (1, ))
    assert_size_stride(primals_48, (95, ), (1, ))
    assert_size_stride(primals_49, (570, ), (1, ))
    assert_size_stride(primals_50, (570, ), (1, ))
    assert_size_stride(primals_51, (570, ), (1, ))
    assert_size_stride(primals_52, (570, ), (1, ))
    assert_size_stride(primals_53, (106, ), (1, ))
    assert_size_stride(primals_54, (106, ), (1, ))
    assert_size_stride(primals_55, (636, ), (1, ))
    assert_size_stride(primals_56, (636, ), (1, ))
    assert_size_stride(primals_57, (636, ), (1, ))
    assert_size_stride(primals_58, (636, ), (1, ))
    assert_size_stride(primals_59, (117, ), (1, ))
    assert_size_stride(primals_60, (117, ), (1, ))
    assert_size_stride(primals_61, (702, ), (1, ))
    assert_size_stride(primals_62, (702, ), (1, ))
    assert_size_stride(primals_63, (702, ), (1, ))
    assert_size_stride(primals_64, (702, ), (1, ))
    assert_size_stride(primals_65, (128, ), (1, ))
    assert_size_stride(primals_66, (128, ), (1, ))
    assert_size_stride(primals_67, (768, ), (1, ))
    assert_size_stride(primals_68, (768, ), (1, ))
    assert_size_stride(primals_69, (768, ), (1, ))
    assert_size_stride(primals_70, (768, ), (1, ))
    assert_size_stride(primals_71, (140, ), (1, ))
    assert_size_stride(primals_72, (140, ), (1, ))
    assert_size_stride(primals_73, (840, ), (1, ))
    assert_size_stride(primals_74, (840, ), (1, ))
    assert_size_stride(primals_75, (840, ), (1, ))
    assert_size_stride(primals_76, (840, ), (1, ))
    assert_size_stride(primals_77, (151, ), (1, ))
    assert_size_stride(primals_78, (151, ), (1, ))
    assert_size_stride(primals_79, (906, ), (1, ))
    assert_size_stride(primals_80, (906, ), (1, ))
    assert_size_stride(primals_81, (906, ), (1, ))
    assert_size_stride(primals_82, (906, ), (1, ))
    assert_size_stride(primals_83, (162, ), (1, ))
    assert_size_stride(primals_84, (162, ), (1, ))
    assert_size_stride(primals_85, (972, ), (1, ))
    assert_size_stride(primals_86, (972, ), (1, ))
    assert_size_stride(primals_87, (972, ), (1, ))
    assert_size_stride(primals_88, (972, ), (1, ))
    assert_size_stride(primals_89, (174, ), (1, ))
    assert_size_stride(primals_90, (174, ), (1, ))
    assert_size_stride(primals_91, (1044, ), (1, ))
    assert_size_stride(primals_92, (1044, ), (1, ))
    assert_size_stride(primals_93, (1044, ), (1, ))
    assert_size_stride(primals_94, (1044, ), (1, ))
    assert_size_stride(primals_95, (185, ), (1, ))
    assert_size_stride(primals_96, (185, ), (1, ))
    assert_size_stride(primals_97, (1280, ), (1, ))
    assert_size_stride(primals_98, (1280, ), (1, ))
    assert_size_stride(primals_99, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_100, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_101, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_102, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_103, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_104, (27, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_105, (162, 27, 1, 1), (27, 1, 1, 1))
    assert_size_stride(primals_106, (162, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_107, (38, 162, 1, 1), (162, 1, 1, 1))
    assert_size_stride(primals_108, (228, 38, 1, 1), (38, 1, 1, 1))
    assert_size_stride(primals_109, (228, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_110, (19, 228, 1, 1), (228, 1, 1, 1))
    assert_size_stride(primals_111, (19, ), (1, ))
    assert_size_stride(primals_112, (19, ), (1, ))
    assert_size_stride(primals_113, (19, ), (1, ))
    assert_size_stride(primals_114, (228, 19, 1, 1), (19, 1, 1, 1))
    assert_size_stride(primals_115, (228, ), (1, ))
    assert_size_stride(primals_116, (50, 228, 1, 1), (228, 1, 1, 1))
    assert_size_stride(primals_117, (300, 50, 1, 1), (50, 1, 1, 1))
    assert_size_stride(primals_118, (300, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_119, (25, 300, 1, 1), (300, 1, 1, 1))
    assert_size_stride(primals_120, (25, ), (1, ))
    assert_size_stride(primals_121, (25, ), (1, ))
    assert_size_stride(primals_122, (25, ), (1, ))
    assert_size_stride(primals_123, (300, 25, 1, 1), (25, 1, 1, 1))
    assert_size_stride(primals_124, (300, ), (1, ))
    assert_size_stride(primals_125, (61, 300, 1, 1), (300, 1, 1, 1))
    assert_size_stride(primals_126, (366, 61, 1, 1), (61, 1, 1, 1))
    assert_size_stride(primals_127, (366, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_128, (30, 366, 1, 1), (366, 1, 1, 1))
    assert_size_stride(primals_129, (30, ), (1, ))
    assert_size_stride(primals_130, (30, ), (1, ))
    assert_size_stride(primals_131, (30, ), (1, ))
    assert_size_stride(primals_132, (366, 30, 1, 1), (30, 1, 1, 1))
    assert_size_stride(primals_133, (366, ), (1, ))
    assert_size_stride(primals_134, (72, 366, 1, 1), (366, 1, 1, 1))
    assert_size_stride(primals_135, (432, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_136, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_137, (36, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(primals_138, (36, ), (1, ))
    assert_size_stride(primals_139, (36, ), (1, ))
    assert_size_stride(primals_140, (36, ), (1, ))
    assert_size_stride(primals_141, (432, 36, 1, 1), (36, 1, 1, 1))
    assert_size_stride(primals_142, (432, ), (1, ))
    assert_size_stride(primals_143, (84, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(primals_144, (504, 84, 1, 1), (84, 1, 1, 1))
    assert_size_stride(primals_145, (504, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_146, (42, 504, 1, 1), (504, 1, 1, 1))
    assert_size_stride(primals_147, (42, ), (1, ))
    assert_size_stride(primals_148, (42, ), (1, ))
    assert_size_stride(primals_149, (42, ), (1, ))
    assert_size_stride(primals_150, (504, 42, 1, 1), (42, 1, 1, 1))
    assert_size_stride(primals_151, (504, ), (1, ))
    assert_size_stride(primals_152, (95, 504, 1, 1), (504, 1, 1, 1))
    assert_size_stride(primals_153, (570, 95, 1, 1), (95, 1, 1, 1))
    assert_size_stride(primals_154, (570, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_155, (47, 570, 1, 1), (570, 1, 1, 1))
    assert_size_stride(primals_156, (47, ), (1, ))
    assert_size_stride(primals_157, (47, ), (1, ))
    assert_size_stride(primals_158, (47, ), (1, ))
    assert_size_stride(primals_159, (570, 47, 1, 1), (47, 1, 1, 1))
    assert_size_stride(primals_160, (570, ), (1, ))
    assert_size_stride(primals_161, (106, 570, 1, 1), (570, 1, 1, 1))
    assert_size_stride(primals_162, (636, 106, 1, 1), (106, 1, 1, 1))
    assert_size_stride(primals_163, (636, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_164, (53, 636, 1, 1), (636, 1, 1, 1))
    assert_size_stride(primals_165, (53, ), (1, ))
    assert_size_stride(primals_166, (53, ), (1, ))
    assert_size_stride(primals_167, (53, ), (1, ))
    assert_size_stride(primals_168, (636, 53, 1, 1), (53, 1, 1, 1))
    assert_size_stride(primals_169, (636, ), (1, ))
    assert_size_stride(primals_170, (117, 636, 1, 1), (636, 1, 1, 1))
    assert_size_stride(primals_171, (702, 117, 1, 1), (117, 1, 1, 1))
    assert_size_stride(primals_172, (702, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_173, (58, 702, 1, 1), (702, 1, 1, 1))
    assert_size_stride(primals_174, (58, ), (1, ))
    assert_size_stride(primals_175, (58, ), (1, ))
    assert_size_stride(primals_176, (58, ), (1, ))
    assert_size_stride(primals_177, (702, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(primals_178, (702, ), (1, ))
    assert_size_stride(primals_179, (128, 702, 1, 1), (702, 1, 1, 1))
    assert_size_stride(primals_180, (768, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_181, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_182, (64, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_183, (64, ), (1, ))
    assert_size_stride(primals_184, (64, ), (1, ))
    assert_size_stride(primals_185, (64, ), (1, ))
    assert_size_stride(primals_186, (768, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_187, (768, ), (1, ))
    assert_size_stride(primals_188, (140, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_189, (840, 140, 1, 1), (140, 1, 1, 1))
    assert_size_stride(primals_190, (840, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_191, (70, 840, 1, 1), (840, 1, 1, 1))
    assert_size_stride(primals_192, (70, ), (1, ))
    assert_size_stride(primals_193, (70, ), (1, ))
    assert_size_stride(primals_194, (70, ), (1, ))
    assert_size_stride(primals_195, (840, 70, 1, 1), (70, 1, 1, 1))
    assert_size_stride(primals_196, (840, ), (1, ))
    assert_size_stride(primals_197, (151, 840, 1, 1), (840, 1, 1, 1))
    assert_size_stride(primals_198, (906, 151, 1, 1), (151, 1, 1, 1))
    assert_size_stride(primals_199, (906, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_200, (75, 906, 1, 1), (906, 1, 1, 1))
    assert_size_stride(primals_201, (75, ), (1, ))
    assert_size_stride(primals_202, (75, ), (1, ))
    assert_size_stride(primals_203, (75, ), (1, ))
    assert_size_stride(primals_204, (906, 75, 1, 1), (75, 1, 1, 1))
    assert_size_stride(primals_205, (906, ), (1, ))
    assert_size_stride(primals_206, (162, 906, 1, 1), (906, 1, 1, 1))
    assert_size_stride(primals_207, (972, 162, 1, 1), (162, 1, 1, 1))
    assert_size_stride(primals_208, (972, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_209, (81, 972, 1, 1), (972, 1, 1, 1))
    assert_size_stride(primals_210, (81, ), (1, ))
    assert_size_stride(primals_211, (81, ), (1, ))
    assert_size_stride(primals_212, (81, ), (1, ))
    assert_size_stride(primals_213, (972, 81, 1, 1), (81, 1, 1, 1))
    assert_size_stride(primals_214, (972, ), (1, ))
    assert_size_stride(primals_215, (174, 972, 1, 1), (972, 1, 1, 1))
    assert_size_stride(primals_216, (1044, 174, 1, 1), (174, 1, 1, 1))
    assert_size_stride(primals_217, (1044, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_218, (87, 1044, 1, 1), (1044, 1, 1, 1))
    assert_size_stride(primals_219, (87, ), (1, ))
    assert_size_stride(primals_220, (87, ), (1, ))
    assert_size_stride(primals_221, (87, ), (1, ))
    assert_size_stride(primals_222, (1044, 87, 1, 1), (87, 1, 1, 1))
    assert_size_stride(primals_223, (1044, ), (1, ))
    assert_size_stride(primals_224, (185, 1044, 1, 1), (1044, 1, 1, 1))
    assert_size_stride(primals_225, (1280, 185, 1, 1), (185, 1, 1, 1))
    assert_size_stride(primals_226, (1000, 1280), (1280, 1))
    assert_size_stride(primals_227, (1000, ), (1, ))
    assert_size_stride(primals_228, (), ())
    assert_size_stride(primals_229, (32, ), (1, ))
    assert_size_stride(primals_230, (32, ), (1, ))
    assert_size_stride(primals_231, (), ())
    assert_size_stride(primals_232, (32, ), (1, ))
    assert_size_stride(primals_233, (32, ), (1, ))
    assert_size_stride(primals_234, (), ())
    assert_size_stride(primals_235, (16, ), (1, ))
    assert_size_stride(primals_236, (16, ), (1, ))
    assert_size_stride(primals_237, (), ())
    assert_size_stride(primals_238, (96, ), (1, ))
    assert_size_stride(primals_239, (96, ), (1, ))
    assert_size_stride(primals_240, (), ())
    assert_size_stride(primals_241, (96, ), (1, ))
    assert_size_stride(primals_242, (96, ), (1, ))
    assert_size_stride(primals_243, (), ())
    assert_size_stride(primals_244, (27, ), (1, ))
    assert_size_stride(primals_245, (27, ), (1, ))
    assert_size_stride(primals_246, (), ())
    assert_size_stride(primals_247, (162, ), (1, ))
    assert_size_stride(primals_248, (162, ), (1, ))
    assert_size_stride(primals_249, (), ())
    assert_size_stride(primals_250, (162, ), (1, ))
    assert_size_stride(primals_251, (162, ), (1, ))
    assert_size_stride(primals_252, (), ())
    assert_size_stride(primals_253, (38, ), (1, ))
    assert_size_stride(primals_254, (38, ), (1, ))
    assert_size_stride(primals_255, (), ())
    assert_size_stride(primals_256, (228, ), (1, ))
    assert_size_stride(primals_257, (228, ), (1, ))
    assert_size_stride(primals_258, (), ())
    assert_size_stride(primals_259, (228, ), (1, ))
    assert_size_stride(primals_260, (228, ), (1, ))
    assert_size_stride(primals_261, (), ())
    assert_size_stride(primals_262, (50, ), (1, ))
    assert_size_stride(primals_263, (50, ), (1, ))
    assert_size_stride(primals_264, (), ())
    assert_size_stride(primals_265, (300, ), (1, ))
    assert_size_stride(primals_266, (300, ), (1, ))
    assert_size_stride(primals_267, (), ())
    assert_size_stride(primals_268, (300, ), (1, ))
    assert_size_stride(primals_269, (300, ), (1, ))
    assert_size_stride(primals_270, (), ())
    assert_size_stride(primals_271, (61, ), (1, ))
    assert_size_stride(primals_272, (61, ), (1, ))
    assert_size_stride(primals_273, (), ())
    assert_size_stride(primals_274, (366, ), (1, ))
    assert_size_stride(primals_275, (366, ), (1, ))
    assert_size_stride(primals_276, (), ())
    assert_size_stride(primals_277, (366, ), (1, ))
    assert_size_stride(primals_278, (366, ), (1, ))
    assert_size_stride(primals_279, (), ())
    assert_size_stride(primals_280, (72, ), (1, ))
    assert_size_stride(primals_281, (72, ), (1, ))
    assert_size_stride(primals_282, (), ())
    assert_size_stride(primals_283, (432, ), (1, ))
    assert_size_stride(primals_284, (432, ), (1, ))
    assert_size_stride(primals_285, (), ())
    assert_size_stride(primals_286, (432, ), (1, ))
    assert_size_stride(primals_287, (432, ), (1, ))
    assert_size_stride(primals_288, (), ())
    assert_size_stride(primals_289, (84, ), (1, ))
    assert_size_stride(primals_290, (84, ), (1, ))
    assert_size_stride(primals_291, (), ())
    assert_size_stride(primals_292, (504, ), (1, ))
    assert_size_stride(primals_293, (504, ), (1, ))
    assert_size_stride(primals_294, (), ())
    assert_size_stride(primals_295, (504, ), (1, ))
    assert_size_stride(primals_296, (504, ), (1, ))
    assert_size_stride(primals_297, (), ())
    assert_size_stride(primals_298, (95, ), (1, ))
    assert_size_stride(primals_299, (95, ), (1, ))
    assert_size_stride(primals_300, (), ())
    assert_size_stride(primals_301, (570, ), (1, ))
    assert_size_stride(primals_302, (570, ), (1, ))
    assert_size_stride(primals_303, (), ())
    assert_size_stride(primals_304, (570, ), (1, ))
    assert_size_stride(primals_305, (570, ), (1, ))
    assert_size_stride(primals_306, (), ())
    assert_size_stride(primals_307, (106, ), (1, ))
    assert_size_stride(primals_308, (106, ), (1, ))
    assert_size_stride(primals_309, (), ())
    assert_size_stride(primals_310, (636, ), (1, ))
    assert_size_stride(primals_311, (636, ), (1, ))
    assert_size_stride(primals_312, (), ())
    assert_size_stride(primals_313, (636, ), (1, ))
    assert_size_stride(primals_314, (636, ), (1, ))
    assert_size_stride(primals_315, (), ())
    assert_size_stride(primals_316, (117, ), (1, ))
    assert_size_stride(primals_317, (117, ), (1, ))
    assert_size_stride(primals_318, (), ())
    assert_size_stride(primals_319, (702, ), (1, ))
    assert_size_stride(primals_320, (702, ), (1, ))
    assert_size_stride(primals_321, (), ())
    assert_size_stride(primals_322, (702, ), (1, ))
    assert_size_stride(primals_323, (702, ), (1, ))
    assert_size_stride(primals_324, (), ())
    assert_size_stride(primals_325, (128, ), (1, ))
    assert_size_stride(primals_326, (128, ), (1, ))
    assert_size_stride(primals_327, (), ())
    assert_size_stride(primals_328, (768, ), (1, ))
    assert_size_stride(primals_329, (768, ), (1, ))
    assert_size_stride(primals_330, (), ())
    assert_size_stride(primals_331, (768, ), (1, ))
    assert_size_stride(primals_332, (768, ), (1, ))
    assert_size_stride(primals_333, (), ())
    assert_size_stride(primals_334, (140, ), (1, ))
    assert_size_stride(primals_335, (140, ), (1, ))
    assert_size_stride(primals_336, (), ())
    assert_size_stride(primals_337, (840, ), (1, ))
    assert_size_stride(primals_338, (840, ), (1, ))
    assert_size_stride(primals_339, (), ())
    assert_size_stride(primals_340, (840, ), (1, ))
    assert_size_stride(primals_341, (840, ), (1, ))
    assert_size_stride(primals_342, (), ())
    assert_size_stride(primals_343, (151, ), (1, ))
    assert_size_stride(primals_344, (151, ), (1, ))
    assert_size_stride(primals_345, (), ())
    assert_size_stride(primals_346, (906, ), (1, ))
    assert_size_stride(primals_347, (906, ), (1, ))
    assert_size_stride(primals_348, (), ())
    assert_size_stride(primals_349, (906, ), (1, ))
    assert_size_stride(primals_350, (906, ), (1, ))
    assert_size_stride(primals_351, (), ())
    assert_size_stride(primals_352, (162, ), (1, ))
    assert_size_stride(primals_353, (162, ), (1, ))
    assert_size_stride(primals_354, (), ())
    assert_size_stride(primals_355, (972, ), (1, ))
    assert_size_stride(primals_356, (972, ), (1, ))
    assert_size_stride(primals_357, (), ())
    assert_size_stride(primals_358, (972, ), (1, ))
    assert_size_stride(primals_359, (972, ), (1, ))
    assert_size_stride(primals_360, (), ())
    assert_size_stride(primals_361, (174, ), (1, ))
    assert_size_stride(primals_362, (174, ), (1, ))
    assert_size_stride(primals_363, (), ())
    assert_size_stride(primals_364, (1044, ), (1, ))
    assert_size_stride(primals_365, (1044, ), (1, ))
    assert_size_stride(primals_366, (), ())
    assert_size_stride(primals_367, (1044, ), (1, ))
    assert_size_stride(primals_368, (1044, ), (1, ))
    assert_size_stride(primals_369, (), ())
    assert_size_stride(primals_370, (185, ), (1, ))
    assert_size_stride(primals_371, (185, ), (1, ))
    assert_size_stride(primals_372, (), ())
    assert_size_stride(primals_373, (1280, ), (1, ))
    assert_size_stride(primals_374, (1280, ), (1, ))
    assert_size_stride(primals_375, (19, ), (1, ))
    assert_size_stride(primals_376, (19, ), (1, ))
    assert_size_stride(primals_377, (), ())
    assert_size_stride(primals_378, (25, ), (1, ))
    assert_size_stride(primals_379, (25, ), (1, ))
    assert_size_stride(primals_380, (), ())
    assert_size_stride(primals_381, (30, ), (1, ))
    assert_size_stride(primals_382, (30, ), (1, ))
    assert_size_stride(primals_383, (), ())
    assert_size_stride(primals_384, (36, ), (1, ))
    assert_size_stride(primals_385, (36, ), (1, ))
    assert_size_stride(primals_386, (), ())
    assert_size_stride(primals_387, (42, ), (1, ))
    assert_size_stride(primals_388, (42, ), (1, ))
    assert_size_stride(primals_389, (), ())
    assert_size_stride(primals_390, (47, ), (1, ))
    assert_size_stride(primals_391, (47, ), (1, ))
    assert_size_stride(primals_392, (), ())
    assert_size_stride(primals_393, (53, ), (1, ))
    assert_size_stride(primals_394, (53, ), (1, ))
    assert_size_stride(primals_395, (), ())
    assert_size_stride(primals_396, (58, ), (1, ))
    assert_size_stride(primals_397, (58, ), (1, ))
    assert_size_stride(primals_398, (), ())
    assert_size_stride(primals_399, (64, ), (1, ))
    assert_size_stride(primals_400, (64, ), (1, ))
    assert_size_stride(primals_401, (), ())
    assert_size_stride(primals_402, (70, ), (1, ))
    assert_size_stride(primals_403, (70, ), (1, ))
    assert_size_stride(primals_404, (), ())
    assert_size_stride(primals_405, (75, ), (1, ))
    assert_size_stride(primals_406, (75, ), (1, ))
    assert_size_stride(primals_407, (), ())
    assert_size_stride(primals_408, (81, ), (1, ))
    assert_size_stride(primals_409, (81, ), (1, ))
    assert_size_stride(primals_410, (), ())
    assert_size_stride(primals_411, (87, ), (1, ))
    assert_size_stride(primals_412, (87, ), (1, ))
    assert_size_stride(primals_413, (), ())
    assert_size_stride(primals_414, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    cpp_fused_0(c_void_p(primals_99.data_ptr()), c_void_p(primals_414.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del primals_414
    del primals_99
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf1, buf0, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf2, (8, 32, 112, 112), (401408, 1, 3584, 32))
    buf3 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    buf6 = empty((32, ), device='cpu', dtype=torch.float32)
    buf7 = empty_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    buf8 = empty_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_1(c_void_p(buf2.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()))
    del primals_2
    # Source Nodes: [x_6], Original ATen: [aten.convolution]
    buf9 = extern_kernels.convolution(buf8, primals_100, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf9, (8, 32, 112, 112), (401408, 1, 3584, 32))
    buf10 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    buf11 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    buf13 = empty((32, ), device='cpu', dtype=torch.float32)
    buf14 = empty_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    buf15 = empty_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_hardtanh_2(c_void_p(buf9.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()))
    del primals_4
    # Source Nodes: [x_13], Original ATen: [aten.convolution]
    buf16 = extern_kernels.convolution(buf15, primals_101, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf16, (8, 16, 112, 112), (200704, 1, 1792, 16))
    buf17 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cpu', dtype=torch.float32)
    buf18 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cpu', dtype=torch.float32)
    buf20 = empty((16, ), device='cpu', dtype=torch.float32)
    buf21 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_3(c_void_p(buf16.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()))
    del primals_6
    # Source Nodes: [x_19], Original ATen: [aten.convolution]
    buf22 = extern_kernels.convolution(buf21, primals_102, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf22, (8, 96, 112, 112), (1204224, 1, 10752, 96))
    buf23 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cpu', dtype=torch.float32)
    buf24 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cpu', dtype=torch.float32)
    buf26 = empty((96, ), device='cpu', dtype=torch.float32)
    buf27 = empty_strided((8, 96, 112, 112), (1204224, 1, 10752, 96), device='cpu', dtype=torch.float32)
    buf28 = empty_strided((8, 96, 112, 112), (1204224, 1, 10752, 96), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_4(c_void_p(buf22.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()))
    del primals_8
    # Source Nodes: [x_25], Original ATen: [aten.convolution]
    buf29 = extern_kernels.convolution(buf28, primals_103, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
    assert_size_stride(buf29, (8, 96, 56, 56), (301056, 1, 5376, 96))
    buf30 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cpu', dtype=torch.float32)
    buf31 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cpu', dtype=torch.float32)
    buf33 = empty((96, ), device='cpu', dtype=torch.float32)
    buf34 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cpu', dtype=torch.float32)
    buf35 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_hardtanh_5(c_void_p(buf29.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()))
    del primals_10
    # Source Nodes: [x_32], Original ATen: [aten.convolution]
    buf36 = extern_kernels.convolution(buf35, primals_104, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf36, (8, 27, 56, 56), (84672, 1, 1512, 27))
    buf37 = empty_strided((1, 27, 1, 1), (27, 1, 27, 27), device='cpu', dtype=torch.float32)
    buf38 = empty_strided((1, 27, 1, 1), (27, 1, 27, 27), device='cpu', dtype=torch.float32)
    buf40 = empty((27, ), device='cpu', dtype=torch.float32)
    buf41 = empty_strided((8, 27, 56, 56), (84672, 1, 1512, 27), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_6(c_void_p(buf36.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf41.data_ptr()))
    del primals_12
    # Source Nodes: [x_38], Original ATen: [aten.convolution]
    buf42 = extern_kernels.convolution(buf41, primals_105, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf42, (8, 162, 56, 56), (508032, 1, 9072, 162))
    buf43 = empty_strided((1, 162, 1, 1), (162, 1, 162, 162), device='cpu', dtype=torch.float32)
    buf44 = empty_strided((1, 162, 1, 1), (162, 1, 162, 162), device='cpu', dtype=torch.float32)
    buf46 = empty((162, ), device='cpu', dtype=torch.float32)
    buf47 = empty_strided((8, 162, 56, 56), (508032, 1, 9072, 162), device='cpu', dtype=torch.float32)
    buf48 = empty_strided((8, 162, 56, 56), (508032, 1, 9072, 162), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_7(c_void_p(buf42.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()))
    del primals_14
    # Source Nodes: [x_44], Original ATen: [aten.convolution]
    buf49 = extern_kernels.convolution(buf48, primals_106, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=162, bias=None)
    assert_size_stride(buf49, (8, 162, 56, 56), (508032, 1, 9072, 162))
    buf50 = empty_strided((1, 162, 1, 1), (162, 1, 162, 162), device='cpu', dtype=torch.float32)
    buf51 = empty_strided((1, 162, 1, 1), (162, 1, 162, 162), device='cpu', dtype=torch.float32)
    buf53 = empty((162, ), device='cpu', dtype=torch.float32)
    buf54 = empty_strided((8, 162, 56, 56), (508032, 1, 9072, 162), device='cpu', dtype=torch.float32)
    buf55 = empty_strided((8, 162, 56, 56), (508032, 1, 9072, 162), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_hardtanh_8(c_void_p(buf49.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()))
    del primals_16
    # Source Nodes: [x_51], Original ATen: [aten.convolution]
    buf56 = extern_kernels.convolution(buf55, primals_107, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf56, (8, 38, 56, 56), (119168, 1, 2128, 38))
    buf57 = empty_strided((1, 38, 1, 1), (38, 1, 38, 38), device='cpu', dtype=torch.float32)
    buf58 = empty_strided((1, 38, 1, 1), (38, 1, 38, 38), device='cpu', dtype=torch.float32)
    buf60 = empty((38, ), device='cpu', dtype=torch.float32)
    buf61 = empty_strided((8, 38, 56, 56), (119168, 1, 2128, 38), device='cpu', dtype=torch.float32)
    buf62 = buf61; del buf61  # reuse
    cpp_fused__native_batch_norm_legit_functional_cat_9(c_void_p(buf62.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf60.data_ptr()))
    del primals_18
    # Source Nodes: [x_58], Original ATen: [aten.convolution]
    buf63 = extern_kernels.convolution(buf62, primals_108, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf63, (8, 228, 56, 56), (715008, 1, 12768, 228))
    buf64 = empty_strided((1, 228, 1, 1), (228, 1, 228, 228), device='cpu', dtype=torch.float32)
    buf65 = empty_strided((1, 228, 1, 1), (228, 1, 228, 228), device='cpu', dtype=torch.float32)
    buf67 = empty((228, ), device='cpu', dtype=torch.float32)
    buf68 = empty_strided((8, 228, 56, 56), (715008, 1, 12768, 228), device='cpu', dtype=torch.float32)
    buf69 = empty_strided((8, 228, 56, 56), (715008, 1, 12768, 228), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_10(c_void_p(buf63.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()))
    del primals_20
    # Source Nodes: [x_64], Original ATen: [aten.convolution]
    buf70 = extern_kernels.convolution(buf69, primals_109, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=228, bias=None)
    assert_size_stride(buf70, (8, 228, 28, 28), (178752, 1, 6384, 228))
    buf71 = empty_strided((1, 228, 1, 1), (228, 1, 228, 228), device='cpu', dtype=torch.float32)
    buf72 = empty_strided((1, 228, 1, 1), (228, 1, 228, 228), device='cpu', dtype=torch.float32)
    buf74 = empty((228, ), device='cpu', dtype=torch.float32)
    buf75 = empty_strided((8, 228, 28, 28), (178752, 1, 6384, 228), device='cpu', dtype=torch.float32)
    buf76 = empty_strided((8, 228, 1, 1), (228, 1, 1824, 1824), device='cpu', dtype=torch.float32)
    buf77 = reinterpret_tensor(buf76, (8, 228, 1, 1), (228, 1, 228, 228), 0); del buf76  # reuse
    cpp_fused__native_batch_norm_legit_functional_mean_11(c_void_p(buf77.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf75.data_ptr()))
    del primals_22
    # Source Nodes: [x_se_1], Original ATen: [aten.convolution]
    buf78 = extern_kernels.convolution(buf77, primals_110, primals_111, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf78, (8, 19, 1, 1), (19, 1, 19, 19))
    del primals_111
    buf79 = empty_strided((1, 19, 1, 1), (19, 1, 19, 19), device='cpu', dtype=torch.float32)
    buf80 = empty_strided((1, 19, 1, 1), (19, 1, 19, 19), device='cpu', dtype=torch.float32)
    buf82 = empty_strided((8, 19, 1, 1), (19, 1, 19, 19), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_12(c_void_p(buf78.data_ptr()), c_void_p(primals_112.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf82.data_ptr()))
    del primals_113
    # Source Nodes: [x_se_3], Original ATen: [aten.convolution]
    buf83 = extern_kernels.convolution(buf82, primals_114, primals_115, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf83, (8, 228, 1, 1), (228, 1, 228, 228))
    del primals_115
    buf84 = empty_strided((8, 228, 28, 28), (178752, 1, 6384, 228), device='cpu', dtype=torch.float32)
    cpp_fused_hardtanh_mul_sigmoid_13(c_void_p(buf75.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()))
    # Source Nodes: [x_72], Original ATen: [aten.convolution]
    buf85 = extern_kernels.convolution(buf84, primals_116, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf85, (8, 50, 28, 28), (39200, 1, 1400, 50))
    buf86 = empty_strided((1, 50, 1, 1), (50, 1, 50, 50), device='cpu', dtype=torch.float32)
    buf87 = empty_strided((1, 50, 1, 1), (50, 1, 50, 50), device='cpu', dtype=torch.float32)
    buf89 = empty((50, ), device='cpu', dtype=torch.float32)
    buf90 = empty_strided((8, 50, 28, 28), (39200, 1, 1400, 50), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_14(c_void_p(buf85.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf90.data_ptr()))
    del primals_24
    # Source Nodes: [x_78], Original ATen: [aten.convolution]
    buf91 = extern_kernels.convolution(buf90, primals_117, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf91, (8, 300, 28, 28), (235200, 1, 8400, 300))
    buf92 = empty_strided((1, 300, 1, 1), (300, 1, 300, 300), device='cpu', dtype=torch.float32)
    buf93 = empty_strided((1, 300, 1, 1), (300, 1, 300, 300), device='cpu', dtype=torch.float32)
    buf95 = empty((300, ), device='cpu', dtype=torch.float32)
    buf96 = empty_strided((8, 300, 28, 28), (235200, 1, 8400, 300), device='cpu', dtype=torch.float32)
    buf97 = empty_strided((8, 300, 28, 28), (235200, 1, 8400, 300), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_15(c_void_p(buf91.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf97.data_ptr()))
    del primals_26
    # Source Nodes: [x_84], Original ATen: [aten.convolution]
    buf98 = extern_kernels.convolution(buf97, primals_118, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=300, bias=None)
    assert_size_stride(buf98, (8, 300, 28, 28), (235200, 1, 8400, 300))
    buf99 = empty_strided((1, 300, 1, 1), (300, 1, 300, 300), device='cpu', dtype=torch.float32)
    buf100 = empty_strided((1, 300, 1, 1), (300, 1, 300, 300), device='cpu', dtype=torch.float32)
    buf102 = empty((300, ), device='cpu', dtype=torch.float32)
    buf103 = empty_strided((8, 300, 28, 28), (235200, 1, 8400, 300), device='cpu', dtype=torch.float32)
    buf104 = empty_strided((8, 300, 1, 1), (300, 1, 2400, 2400), device='cpu', dtype=torch.float32)
    buf105 = reinterpret_tensor(buf104, (8, 300, 1, 1), (300, 1, 300, 300), 0); del buf104  # reuse
    cpp_fused__native_batch_norm_legit_functional_mean_16(c_void_p(buf105.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()))
    del primals_28
    # Source Nodes: [x_se_5], Original ATen: [aten.convolution]
    buf106 = extern_kernels.convolution(buf105, primals_119, primals_120, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf106, (8, 25, 1, 1), (25, 1, 25, 25))
    del primals_120
    buf107 = empty_strided((1, 25, 1, 1), (25, 1, 25, 25), device='cpu', dtype=torch.float32)
    buf108 = empty_strided((1, 25, 1, 1), (25, 1, 25, 25), device='cpu', dtype=torch.float32)
    buf110 = empty_strided((8, 25, 1, 1), (25, 1, 25, 25), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_17(c_void_p(buf106.data_ptr()), c_void_p(primals_121.data_ptr()), c_void_p(primals_122.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf110.data_ptr()))
    del primals_122
    # Source Nodes: [x_se_7], Original ATen: [aten.convolution]
    buf111 = extern_kernels.convolution(buf110, primals_123, primals_124, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf111, (8, 300, 1, 1), (300, 1, 300, 300))
    del primals_124
    buf112 = empty_strided((8, 300, 28, 28), (235200, 1, 8400, 300), device='cpu', dtype=torch.float32)
    cpp_fused_hardtanh_mul_sigmoid_18(c_void_p(buf103.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf112.data_ptr()))
    # Source Nodes: [x_92], Original ATen: [aten.convolution]
    buf113 = extern_kernels.convolution(buf112, primals_125, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf113, (8, 61, 28, 28), (47824, 1, 1708, 61))
    buf114 = empty_strided((1, 61, 1, 1), (61, 1, 61, 61), device='cpu', dtype=torch.float32)
    buf115 = empty_strided((1, 61, 1, 1), (61, 1, 61, 61), device='cpu', dtype=torch.float32)
    buf117 = empty((61, ), device='cpu', dtype=torch.float32)
    buf118 = empty_strided((8, 61, 28, 28), (47824, 1, 1708, 61), device='cpu', dtype=torch.float32)
    buf119 = buf118; del buf118  # reuse
    cpp_fused__native_batch_norm_legit_functional_cat_19(c_void_p(buf119.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf117.data_ptr()))
    del primals_30
    # Source Nodes: [x_99], Original ATen: [aten.convolution]
    buf120 = extern_kernels.convolution(buf119, primals_126, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf120, (8, 366, 28, 28), (286944, 1, 10248, 366))
    buf121 = empty_strided((1, 366, 1, 1), (366, 1, 366, 366), device='cpu', dtype=torch.float32)
    buf122 = empty_strided((1, 366, 1, 1), (366, 1, 366, 366), device='cpu', dtype=torch.float32)
    buf124 = empty((366, ), device='cpu', dtype=torch.float32)
    buf125 = empty_strided((8, 366, 28, 28), (286944, 1, 10248, 366), device='cpu', dtype=torch.float32)
    buf126 = empty_strided((8, 366, 28, 28), (286944, 1, 10248, 366), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_20(c_void_p(buf120.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf126.data_ptr()))
    del primals_32
    # Source Nodes: [x_105], Original ATen: [aten.convolution]
    buf127 = extern_kernels.convolution(buf126, primals_127, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=366, bias=None)
    assert_size_stride(buf127, (8, 366, 14, 14), (71736, 1, 5124, 366))
    buf128 = empty_strided((1, 366, 1, 1), (366, 1, 366, 366), device='cpu', dtype=torch.float32)
    buf129 = empty_strided((1, 366, 1, 1), (366, 1, 366, 366), device='cpu', dtype=torch.float32)
    buf131 = empty((366, ), device='cpu', dtype=torch.float32)
    buf132 = empty_strided((8, 366, 14, 14), (71736, 1, 5124, 366), device='cpu', dtype=torch.float32)
    buf133 = empty_strided((8, 366, 1, 1), (366, 1, 2928, 2928), device='cpu', dtype=torch.float32)
    buf134 = reinterpret_tensor(buf133, (8, 366, 1, 1), (366, 1, 366, 366), 0); del buf133  # reuse
    cpp_fused__native_batch_norm_legit_functional_mean_21(c_void_p(buf134.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(primals_34.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()))
    del primals_34
    # Source Nodes: [x_se_9], Original ATen: [aten.convolution]
    buf135 = extern_kernels.convolution(buf134, primals_128, primals_129, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf135, (8, 30, 1, 1), (30, 1, 30, 30))
    del primals_129
    buf136 = empty_strided((1, 30, 1, 1), (30, 1, 30, 30), device='cpu', dtype=torch.float32)
    buf137 = empty_strided((1, 30, 1, 1), (30, 1, 30, 30), device='cpu', dtype=torch.float32)
    buf139 = empty_strided((8, 30, 1, 1), (30, 1, 30, 30), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_22(c_void_p(buf135.data_ptr()), c_void_p(primals_130.data_ptr()), c_void_p(primals_131.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf139.data_ptr()))
    del primals_131
    # Source Nodes: [x_se_11], Original ATen: [aten.convolution]
    buf140 = extern_kernels.convolution(buf139, primals_132, primals_133, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf140, (8, 366, 1, 1), (366, 1, 366, 366))
    del primals_133
    buf141 = empty_strided((8, 366, 14, 14), (71736, 1, 5124, 366), device='cpu', dtype=torch.float32)
    cpp_fused_hardtanh_mul_sigmoid_23(c_void_p(buf132.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()))
    # Source Nodes: [x_113], Original ATen: [aten.convolution]
    buf142 = extern_kernels.convolution(buf141, primals_134, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf142, (8, 72, 14, 14), (14112, 1, 1008, 72))
    buf143 = empty_strided((1, 72, 1, 1), (72, 1, 72, 72), device='cpu', dtype=torch.float32)
    buf144 = empty_strided((1, 72, 1, 1), (72, 1, 72, 72), device='cpu', dtype=torch.float32)
    buf146 = empty((72, ), device='cpu', dtype=torch.float32)
    buf147 = empty_strided((8, 72, 14, 14), (14112, 1, 1008, 72), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_24(c_void_p(buf142.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf147.data_ptr()))
    del primals_36
    # Source Nodes: [x_119], Original ATen: [aten.convolution]
    buf148 = extern_kernels.convolution(buf147, primals_135, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf148, (8, 432, 14, 14), (84672, 1, 6048, 432))
    buf149 = empty_strided((1, 432, 1, 1), (432, 1, 432, 432), device='cpu', dtype=torch.float32)
    buf150 = empty_strided((1, 432, 1, 1), (432, 1, 432, 432), device='cpu', dtype=torch.float32)
    buf152 = empty((432, ), device='cpu', dtype=torch.float32)
    buf153 = empty_strided((8, 432, 14, 14), (84672, 1, 6048, 432), device='cpu', dtype=torch.float32)
    buf154 = empty_strided((8, 432, 14, 14), (84672, 1, 6048, 432), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_25(c_void_p(buf148.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()))
    del primals_38
    # Source Nodes: [x_125], Original ATen: [aten.convolution]
    buf155 = extern_kernels.convolution(buf154, primals_136, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
    assert_size_stride(buf155, (8, 432, 14, 14), (84672, 1, 6048, 432))
    buf156 = empty_strided((1, 432, 1, 1), (432, 1, 432, 432), device='cpu', dtype=torch.float32)
    buf157 = empty_strided((1, 432, 1, 1), (432, 1, 432, 432), device='cpu', dtype=torch.float32)
    buf159 = empty((432, ), device='cpu', dtype=torch.float32)
    buf160 = empty_strided((8, 432, 14, 14), (84672, 1, 6048, 432), device='cpu', dtype=torch.float32)
    buf161 = empty_strided((8, 432, 1, 1), (432, 1, 3456, 3456), device='cpu', dtype=torch.float32)
    buf162 = reinterpret_tensor(buf161, (8, 432, 1, 1), (432, 1, 432, 432), 0); del buf161  # reuse
    cpp_fused__native_batch_norm_legit_functional_mean_26(c_void_p(buf162.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(primals_40.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf160.data_ptr()))
    del primals_40
    # Source Nodes: [x_se_13], Original ATen: [aten.convolution]
    buf163 = extern_kernels.convolution(buf162, primals_137, primals_138, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf163, (8, 36, 1, 1), (36, 1, 36, 36))
    del primals_138
    buf164 = empty_strided((1, 36, 1, 1), (36, 1, 36, 36), device='cpu', dtype=torch.float32)
    buf165 = empty_strided((1, 36, 1, 1), (36, 1, 36, 36), device='cpu', dtype=torch.float32)
    buf167 = empty_strided((8, 36, 1, 1), (36, 1, 36, 36), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_27(c_void_p(buf163.data_ptr()), c_void_p(primals_139.data_ptr()), c_void_p(primals_140.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf167.data_ptr()))
    del primals_140
    # Source Nodes: [x_se_15], Original ATen: [aten.convolution]
    buf168 = extern_kernels.convolution(buf167, primals_141, primals_142, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf168, (8, 432, 1, 1), (432, 1, 432, 432))
    del primals_142
    buf169 = empty_strided((8, 432, 14, 14), (84672, 1, 6048, 432), device='cpu', dtype=torch.float32)
    cpp_fused_hardtanh_mul_sigmoid_28(c_void_p(buf160.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()))
    # Source Nodes: [x_133], Original ATen: [aten.convolution]
    buf170 = extern_kernels.convolution(buf169, primals_143, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf170, (8, 84, 14, 14), (16464, 1, 1176, 84))
    buf171 = empty_strided((1, 84, 1, 1), (84, 1, 84, 84), device='cpu', dtype=torch.float32)
    buf172 = empty_strided((1, 84, 1, 1), (84, 1, 84, 84), device='cpu', dtype=torch.float32)
    buf174 = empty((84, ), device='cpu', dtype=torch.float32)
    buf175 = empty_strided((8, 84, 14, 14), (16464, 1, 1176, 84), device='cpu', dtype=torch.float32)
    buf176 = buf175; del buf175  # reuse
    cpp_fused__native_batch_norm_legit_functional_cat_29(c_void_p(buf176.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf174.data_ptr()))
    del primals_42
    # Source Nodes: [x_140], Original ATen: [aten.convolution]
    buf177 = extern_kernels.convolution(buf176, primals_144, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf177, (8, 504, 14, 14), (98784, 1, 7056, 504))
    buf178 = empty_strided((1, 504, 1, 1), (504, 1, 504, 504), device='cpu', dtype=torch.float32)
    buf179 = empty_strided((1, 504, 1, 1), (504, 1, 504, 504), device='cpu', dtype=torch.float32)
    buf181 = empty((504, ), device='cpu', dtype=torch.float32)
    buf182 = empty_strided((8, 504, 14, 14), (98784, 1, 7056, 504), device='cpu', dtype=torch.float32)
    buf183 = empty_strided((8, 504, 14, 14), (98784, 1, 7056, 504), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_30(c_void_p(buf177.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(primals_44.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf183.data_ptr()))
    del primals_44
    # Source Nodes: [x_146], Original ATen: [aten.convolution]
    buf184 = extern_kernels.convolution(buf183, primals_145, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=504, bias=None)
    assert_size_stride(buf184, (8, 504, 14, 14), (98784, 1, 7056, 504))
    buf185 = empty_strided((1, 504, 1, 1), (504, 1, 504, 504), device='cpu', dtype=torch.float32)
    buf186 = empty_strided((1, 504, 1, 1), (504, 1, 504, 504), device='cpu', dtype=torch.float32)
    buf188 = empty((504, ), device='cpu', dtype=torch.float32)
    buf189 = empty_strided((8, 504, 14, 14), (98784, 1, 7056, 504), device='cpu', dtype=torch.float32)
    buf190 = empty_strided((8, 504, 1, 1), (504, 1, 4032, 4032), device='cpu', dtype=torch.float32)
    buf191 = reinterpret_tensor(buf190, (8, 504, 1, 1), (504, 1, 504, 504), 0); del buf190  # reuse
    cpp_fused__native_batch_norm_legit_functional_mean_31(c_void_p(buf191.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(primals_46.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf189.data_ptr()))
    del primals_46
    # Source Nodes: [x_se_17], Original ATen: [aten.convolution]
    buf192 = extern_kernels.convolution(buf191, primals_146, primals_147, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf192, (8, 42, 1, 1), (42, 1, 42, 42))
    del primals_147
    buf193 = empty_strided((1, 42, 1, 1), (42, 1, 42, 42), device='cpu', dtype=torch.float32)
    buf194 = empty_strided((1, 42, 1, 1), (42, 1, 42, 42), device='cpu', dtype=torch.float32)
    buf196 = empty_strided((8, 42, 1, 1), (42, 1, 42, 42), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_32(c_void_p(buf192.data_ptr()), c_void_p(primals_148.data_ptr()), c_void_p(primals_149.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf196.data_ptr()))
    del primals_149
    # Source Nodes: [x_se_19], Original ATen: [aten.convolution]
    buf197 = extern_kernels.convolution(buf196, primals_150, primals_151, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf197, (8, 504, 1, 1), (504, 1, 504, 504))
    del primals_151
    buf198 = empty_strided((8, 504, 14, 14), (98784, 1, 7056, 504), device='cpu', dtype=torch.float32)
    cpp_fused_hardtanh_mul_sigmoid_33(c_void_p(buf189.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf198.data_ptr()))
    # Source Nodes: [x_154], Original ATen: [aten.convolution]
    buf199 = extern_kernels.convolution(buf198, primals_152, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf199, (8, 95, 14, 14), (18620, 1, 1330, 95))
    buf200 = empty_strided((1, 95, 1, 1), (95, 1, 95, 95), device='cpu', dtype=torch.float32)
    buf201 = empty_strided((1, 95, 1, 1), (95, 1, 95, 95), device='cpu', dtype=torch.float32)
    buf203 = empty((95, ), device='cpu', dtype=torch.float32)
    buf204 = empty_strided((8, 95, 14, 14), (18620, 1, 1330, 95), device='cpu', dtype=torch.float32)
    buf205 = buf204; del buf204  # reuse
    cpp_fused__native_batch_norm_legit_functional_cat_34(c_void_p(buf205.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(primals_48.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf203.data_ptr()))
    del primals_48
    # Source Nodes: [x_161], Original ATen: [aten.convolution]
    buf206 = extern_kernels.convolution(buf205, primals_153, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf206, (8, 570, 14, 14), (111720, 1, 7980, 570))
    buf207 = empty_strided((1, 570, 1, 1), (570, 1, 570, 570), device='cpu', dtype=torch.float32)
    buf208 = empty_strided((1, 570, 1, 1), (570, 1, 570, 570), device='cpu', dtype=torch.float32)
    buf210 = empty((570, ), device='cpu', dtype=torch.float32)
    buf211 = empty_strided((8, 570, 14, 14), (111720, 1, 7980, 570), device='cpu', dtype=torch.float32)
    buf212 = empty_strided((8, 570, 14, 14), (111720, 1, 7980, 570), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_35(c_void_p(buf206.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf212.data_ptr()))
    del primals_50
    # Source Nodes: [x_167], Original ATen: [aten.convolution]
    buf213 = extern_kernels.convolution(buf212, primals_154, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=570, bias=None)
    assert_size_stride(buf213, (8, 570, 14, 14), (111720, 1, 7980, 570))
    buf214 = empty_strided((1, 570, 1, 1), (570, 1, 570, 570), device='cpu', dtype=torch.float32)
    buf215 = empty_strided((1, 570, 1, 1), (570, 1, 570, 570), device='cpu', dtype=torch.float32)
    buf217 = empty((570, ), device='cpu', dtype=torch.float32)
    buf218 = empty_strided((8, 570, 14, 14), (111720, 1, 7980, 570), device='cpu', dtype=torch.float32)
    buf219 = empty_strided((8, 570, 1, 1), (570, 1, 4560, 4560), device='cpu', dtype=torch.float32)
    buf220 = reinterpret_tensor(buf219, (8, 570, 1, 1), (570, 1, 570, 570), 0); del buf219  # reuse
    cpp_fused__native_batch_norm_legit_functional_mean_36(c_void_p(buf220.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(primals_52.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf218.data_ptr()))
    del primals_52
    # Source Nodes: [x_se_21], Original ATen: [aten.convolution]
    buf221 = extern_kernels.convolution(buf220, primals_155, primals_156, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf221, (8, 47, 1, 1), (47, 1, 47, 47))
    del primals_156
    buf222 = empty_strided((1, 47, 1, 1), (47, 1, 47, 47), device='cpu', dtype=torch.float32)
    buf223 = empty_strided((1, 47, 1, 1), (47, 1, 47, 47), device='cpu', dtype=torch.float32)
    buf225 = empty_strided((8, 47, 1, 1), (47, 1, 47, 47), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_37(c_void_p(buf221.data_ptr()), c_void_p(primals_157.data_ptr()), c_void_p(primals_158.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf225.data_ptr()))
    del primals_158
    # Source Nodes: [x_se_23], Original ATen: [aten.convolution]
    buf226 = extern_kernels.convolution(buf225, primals_159, primals_160, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf226, (8, 570, 1, 1), (570, 1, 570, 570))
    del primals_160
    buf227 = empty_strided((8, 570, 14, 14), (111720, 1, 7980, 570), device='cpu', dtype=torch.float32)
    cpp_fused_hardtanh_mul_sigmoid_38(c_void_p(buf218.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf227.data_ptr()))
    # Source Nodes: [x_175], Original ATen: [aten.convolution]
    buf228 = extern_kernels.convolution(buf227, primals_161, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf228, (8, 106, 14, 14), (20776, 1, 1484, 106))
    buf229 = empty_strided((1, 106, 1, 1), (106, 1, 106, 106), device='cpu', dtype=torch.float32)
    buf230 = empty_strided((1, 106, 1, 1), (106, 1, 106, 106), device='cpu', dtype=torch.float32)
    buf232 = empty((106, ), device='cpu', dtype=torch.float32)
    buf233 = empty_strided((8, 106, 14, 14), (20776, 1, 1484, 106), device='cpu', dtype=torch.float32)
    buf234 = buf233; del buf233  # reuse
    cpp_fused__native_batch_norm_legit_functional_cat_39(c_void_p(buf234.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf232.data_ptr()))
    del primals_54
    # Source Nodes: [x_182], Original ATen: [aten.convolution]
    buf235 = extern_kernels.convolution(buf234, primals_162, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf235, (8, 636, 14, 14), (124656, 1, 8904, 636))
    buf236 = empty_strided((1, 636, 1, 1), (636, 1, 636, 636), device='cpu', dtype=torch.float32)
    buf237 = empty_strided((1, 636, 1, 1), (636, 1, 636, 636), device='cpu', dtype=torch.float32)
    buf239 = empty((636, ), device='cpu', dtype=torch.float32)
    buf240 = empty_strided((8, 636, 14, 14), (124656, 1, 8904, 636), device='cpu', dtype=torch.float32)
    buf241 = empty_strided((8, 636, 14, 14), (124656, 1, 8904, 636), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_40(c_void_p(buf235.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(primals_56.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf241.data_ptr()))
    del primals_56
    # Source Nodes: [x_188], Original ATen: [aten.convolution]
    buf242 = extern_kernels.convolution(buf241, primals_163, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=636, bias=None)
    assert_size_stride(buf242, (8, 636, 14, 14), (124656, 1, 8904, 636))
    buf243 = empty_strided((1, 636, 1, 1), (636, 1, 636, 636), device='cpu', dtype=torch.float32)
    buf244 = empty_strided((1, 636, 1, 1), (636, 1, 636, 636), device='cpu', dtype=torch.float32)
    buf246 = empty((636, ), device='cpu', dtype=torch.float32)
    buf247 = empty_strided((8, 636, 14, 14), (124656, 1, 8904, 636), device='cpu', dtype=torch.float32)
    buf248 = empty_strided((8, 636, 1, 1), (636, 1, 5088, 5088), device='cpu', dtype=torch.float32)
    buf249 = reinterpret_tensor(buf248, (8, 636, 1, 1), (636, 1, 636, 636), 0); del buf248  # reuse
    cpp_fused__native_batch_norm_legit_functional_mean_41(c_void_p(buf249.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(primals_58.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf247.data_ptr()))
    del primals_58
    # Source Nodes: [x_se_25], Original ATen: [aten.convolution]
    buf250 = extern_kernels.convolution(buf249, primals_164, primals_165, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf250, (8, 53, 1, 1), (53, 1, 53, 53))
    del primals_165
    buf251 = empty_strided((1, 53, 1, 1), (53, 1, 53, 53), device='cpu', dtype=torch.float32)
    buf252 = empty_strided((1, 53, 1, 1), (53, 1, 53, 53), device='cpu', dtype=torch.float32)
    buf254 = empty_strided((8, 53, 1, 1), (53, 1, 53, 53), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_42(c_void_p(buf250.data_ptr()), c_void_p(primals_166.data_ptr()), c_void_p(primals_167.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf254.data_ptr()))
    del primals_167
    # Source Nodes: [x_se_27], Original ATen: [aten.convolution]
    buf255 = extern_kernels.convolution(buf254, primals_168, primals_169, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf255, (8, 636, 1, 1), (636, 1, 636, 636))
    del primals_169
    buf256 = empty_strided((8, 636, 14, 14), (124656, 1, 8904, 636), device='cpu', dtype=torch.float32)
    cpp_fused_hardtanh_mul_sigmoid_43(c_void_p(buf247.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf256.data_ptr()))
    # Source Nodes: [x_196], Original ATen: [aten.convolution]
    buf257 = extern_kernels.convolution(buf256, primals_170, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf257, (8, 117, 14, 14), (22932, 1, 1638, 117))
    buf258 = empty_strided((1, 117, 1, 1), (117, 1, 117, 117), device='cpu', dtype=torch.float32)
    buf259 = empty_strided((1, 117, 1, 1), (117, 1, 117, 117), device='cpu', dtype=torch.float32)
    buf261 = empty((117, ), device='cpu', dtype=torch.float32)
    buf262 = empty_strided((8, 117, 14, 14), (22932, 1, 1638, 117), device='cpu', dtype=torch.float32)
    buf263 = buf262; del buf262  # reuse
    cpp_fused__native_batch_norm_legit_functional_cat_44(c_void_p(buf263.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(primals_60.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf261.data_ptr()))
    del primals_60
    # Source Nodes: [x_203], Original ATen: [aten.convolution]
    buf264 = extern_kernels.convolution(buf263, primals_171, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf264, (8, 702, 14, 14), (137592, 1, 9828, 702))
    buf265 = empty_strided((1, 702, 1, 1), (702, 1, 702, 702), device='cpu', dtype=torch.float32)
    buf266 = empty_strided((1, 702, 1, 1), (702, 1, 702, 702), device='cpu', dtype=torch.float32)
    buf268 = empty((702, ), device='cpu', dtype=torch.float32)
    buf269 = empty_strided((8, 702, 14, 14), (137592, 1, 9828, 702), device='cpu', dtype=torch.float32)
    buf270 = empty_strided((8, 702, 14, 14), (137592, 1, 9828, 702), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_45(c_void_p(buf264.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf270.data_ptr()))
    del primals_62
    # Source Nodes: [x_209], Original ATen: [aten.convolution]
    buf271 = extern_kernels.convolution(buf270, primals_172, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=702, bias=None)
    assert_size_stride(buf271, (8, 702, 14, 14), (137592, 1, 9828, 702))
    buf272 = empty_strided((1, 702, 1, 1), (702, 1, 702, 702), device='cpu', dtype=torch.float32)
    buf273 = empty_strided((1, 702, 1, 1), (702, 1, 702, 702), device='cpu', dtype=torch.float32)
    buf275 = empty((702, ), device='cpu', dtype=torch.float32)
    buf276 = empty_strided((8, 702, 14, 14), (137592, 1, 9828, 702), device='cpu', dtype=torch.float32)
    buf277 = empty_strided((8, 702, 1, 1), (702, 1, 5616, 5616), device='cpu', dtype=torch.float32)
    buf278 = reinterpret_tensor(buf277, (8, 702, 1, 1), (702, 1, 702, 702), 0); del buf277  # reuse
    cpp_fused__native_batch_norm_legit_functional_mean_46(c_void_p(buf278.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(primals_64.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf276.data_ptr()))
    del primals_64
    # Source Nodes: [x_se_29], Original ATen: [aten.convolution]
    buf279 = extern_kernels.convolution(buf278, primals_173, primals_174, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf279, (8, 58, 1, 1), (58, 1, 58, 58))
    del primals_174
    buf280 = empty_strided((1, 58, 1, 1), (58, 1, 58, 58), device='cpu', dtype=torch.float32)
    buf281 = empty_strided((1, 58, 1, 1), (58, 1, 58, 58), device='cpu', dtype=torch.float32)
    buf283 = empty_strided((8, 58, 1, 1), (58, 1, 58, 58), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_47(c_void_p(buf279.data_ptr()), c_void_p(primals_175.data_ptr()), c_void_p(primals_176.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf283.data_ptr()))
    del primals_176
    # Source Nodes: [x_se_31], Original ATen: [aten.convolution]
    buf284 = extern_kernels.convolution(buf283, primals_177, primals_178, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf284, (8, 702, 1, 1), (702, 1, 702, 702))
    del primals_178
    buf285 = empty_strided((8, 702, 14, 14), (137592, 1, 9828, 702), device='cpu', dtype=torch.float32)
    cpp_fused_hardtanh_mul_sigmoid_48(c_void_p(buf276.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf285.data_ptr()))
    # Source Nodes: [x_217], Original ATen: [aten.convolution]
    buf286 = extern_kernels.convolution(buf285, primals_179, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf286, (8, 128, 14, 14), (25088, 1, 1792, 128))
    buf287 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf288 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf290 = empty((128, ), device='cpu', dtype=torch.float32)
    buf291 = empty_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    buf292 = buf291; del buf291  # reuse
    cpp_fused__native_batch_norm_legit_functional_cat_49(c_void_p(buf292.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(primals_66.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf290.data_ptr()))
    del primals_66
    # Source Nodes: [x_224], Original ATen: [aten.convolution]
    buf293 = extern_kernels.convolution(buf292, primals_180, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf293, (8, 768, 14, 14), (150528, 1, 10752, 768))
    buf294 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf295 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf297 = empty((768, ), device='cpu', dtype=torch.float32)
    buf298 = empty_strided((8, 768, 14, 14), (150528, 1, 10752, 768), device='cpu', dtype=torch.float32)
    buf299 = empty_strided((8, 768, 14, 14), (150528, 1, 10752, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_50(c_void_p(buf293.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf299.data_ptr()))
    del primals_68
    # Source Nodes: [x_230], Original ATen: [aten.convolution]
    buf300 = extern_kernels.convolution(buf299, primals_181, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
    assert_size_stride(buf300, (8, 768, 7, 7), (37632, 1, 5376, 768))
    buf301 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf302 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    buf304 = empty((768, ), device='cpu', dtype=torch.float32)
    buf305 = empty_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    buf306 = empty_strided((8, 768, 1, 1), (768, 1, 6144, 6144), device='cpu', dtype=torch.float32)
    buf307 = reinterpret_tensor(buf306, (8, 768, 1, 1), (768, 1, 768, 768), 0); del buf306  # reuse
    cpp_fused__native_batch_norm_legit_functional_mean_51(c_void_p(buf307.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(primals_70.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf305.data_ptr()))
    del primals_70
    # Source Nodes: [x_se_33], Original ATen: [aten.convolution]
    buf308 = extern_kernels.convolution(buf307, primals_182, primals_183, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf308, (8, 64, 1, 1), (64, 1, 64, 64))
    del primals_183
    buf309 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf310 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf312 = empty_strided((8, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_52(c_void_p(buf308.data_ptr()), c_void_p(primals_184.data_ptr()), c_void_p(primals_185.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf312.data_ptr()))
    del primals_185
    # Source Nodes: [x_se_35], Original ATen: [aten.convolution]
    buf313 = extern_kernels.convolution(buf312, primals_186, primals_187, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf313, (8, 768, 1, 1), (768, 1, 768, 768))
    del primals_187
    buf314 = empty_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    cpp_fused_hardtanh_mul_sigmoid_53(c_void_p(buf305.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(buf314.data_ptr()))
    # Source Nodes: [x_238], Original ATen: [aten.convolution]
    buf315 = extern_kernels.convolution(buf314, primals_188, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf315, (8, 140, 7, 7), (6860, 1, 980, 140))
    buf316 = empty_strided((1, 140, 1, 1), (140, 1, 140, 140), device='cpu', dtype=torch.float32)
    buf317 = empty_strided((1, 140, 1, 1), (140, 1, 140, 140), device='cpu', dtype=torch.float32)
    buf319 = empty((140, ), device='cpu', dtype=torch.float32)
    buf320 = empty_strided((8, 140, 7, 7), (6860, 1, 980, 140), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_54(c_void_p(buf315.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(primals_72.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf320.data_ptr()))
    del primals_72
    # Source Nodes: [x_244], Original ATen: [aten.convolution]
    buf321 = extern_kernels.convolution(buf320, primals_189, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf321, (8, 840, 7, 7), (41160, 1, 5880, 840))
    buf322 = empty_strided((1, 840, 1, 1), (840, 1, 840, 840), device='cpu', dtype=torch.float32)
    buf323 = empty_strided((1, 840, 1, 1), (840, 1, 840, 840), device='cpu', dtype=torch.float32)
    buf325 = empty((840, ), device='cpu', dtype=torch.float32)
    buf326 = empty_strided((8, 840, 7, 7), (41160, 1, 5880, 840), device='cpu', dtype=torch.float32)
    buf327 = empty_strided((8, 840, 7, 7), (41160, 1, 5880, 840), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_55(c_void_p(buf321.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(buf327.data_ptr()))
    del primals_74
    # Source Nodes: [x_250], Original ATen: [aten.convolution]
    buf328 = extern_kernels.convolution(buf327, primals_190, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=840, bias=None)
    assert_size_stride(buf328, (8, 840, 7, 7), (41160, 1, 5880, 840))
    buf329 = empty_strided((1, 840, 1, 1), (840, 1, 840, 840), device='cpu', dtype=torch.float32)
    buf330 = empty_strided((1, 840, 1, 1), (840, 1, 840, 840), device='cpu', dtype=torch.float32)
    buf332 = empty((840, ), device='cpu', dtype=torch.float32)
    buf333 = empty_strided((8, 840, 7, 7), (41160, 1, 5880, 840), device='cpu', dtype=torch.float32)
    buf334 = empty_strided((8, 840, 1, 1), (840, 1, 6720, 6720), device='cpu', dtype=torch.float32)
    buf335 = reinterpret_tensor(buf334, (8, 840, 1, 1), (840, 1, 840, 840), 0); del buf334  # reuse
    cpp_fused__native_batch_norm_legit_functional_mean_56(c_void_p(buf335.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(primals_76.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf333.data_ptr()))
    del primals_76
    # Source Nodes: [x_se_37], Original ATen: [aten.convolution]
    buf336 = extern_kernels.convolution(buf335, primals_191, primals_192, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf336, (8, 70, 1, 1), (70, 1, 70, 70))
    del primals_192
    buf337 = empty_strided((1, 70, 1, 1), (70, 1, 70, 70), device='cpu', dtype=torch.float32)
    buf338 = empty_strided((1, 70, 1, 1), (70, 1, 70, 70), device='cpu', dtype=torch.float32)
    buf340 = empty_strided((8, 70, 1, 1), (70, 1, 70, 70), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_57(c_void_p(buf336.data_ptr()), c_void_p(primals_193.data_ptr()), c_void_p(primals_194.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf340.data_ptr()))
    del primals_194
    # Source Nodes: [x_se_39], Original ATen: [aten.convolution]
    buf341 = extern_kernels.convolution(buf340, primals_195, primals_196, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf341, (8, 840, 1, 1), (840, 1, 840, 840))
    del primals_196
    buf342 = empty_strided((8, 840, 7, 7), (41160, 1, 5880, 840), device='cpu', dtype=torch.float32)
    cpp_fused_hardtanh_mul_sigmoid_58(c_void_p(buf333.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf342.data_ptr()))
    # Source Nodes: [x_258], Original ATen: [aten.convolution]
    buf343 = extern_kernels.convolution(buf342, primals_197, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf343, (8, 151, 7, 7), (7399, 1, 1057, 151))
    buf344 = empty_strided((1, 151, 1, 1), (151, 1, 151, 151), device='cpu', dtype=torch.float32)
    buf345 = empty_strided((1, 151, 1, 1), (151, 1, 151, 151), device='cpu', dtype=torch.float32)
    buf347 = empty((151, ), device='cpu', dtype=torch.float32)
    buf348 = empty_strided((8, 151, 7, 7), (7399, 1, 1057, 151), device='cpu', dtype=torch.float32)
    buf349 = buf348; del buf348  # reuse
    cpp_fused__native_batch_norm_legit_functional_cat_59(c_void_p(buf349.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(primals_78.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf347.data_ptr()))
    del primals_78
    # Source Nodes: [x_265], Original ATen: [aten.convolution]
    buf350 = extern_kernels.convolution(buf349, primals_198, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf350, (8, 906, 7, 7), (44394, 1, 6342, 906))
    buf351 = empty_strided((1, 906, 1, 1), (906, 1, 906, 906), device='cpu', dtype=torch.float32)
    buf352 = empty_strided((1, 906, 1, 1), (906, 1, 906, 906), device='cpu', dtype=torch.float32)
    buf354 = empty((906, ), device='cpu', dtype=torch.float32)
    buf355 = empty_strided((8, 906, 7, 7), (44394, 1, 6342, 906), device='cpu', dtype=torch.float32)
    buf356 = empty_strided((8, 906, 7, 7), (44394, 1, 6342, 906), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_silu_60(c_void_p(buf350.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf356.data_ptr()))
    del primals_80
    # Source Nodes: [x_271], Original ATen: [aten.convolution]
    buf357 = extern_kernels.convolution(buf356, primals_199, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=906, bias=None)
    assert_size_stride(buf357, (8, 906, 7, 7), (44394, 1, 6342, 906))
    buf358 = empty_strided((1, 906, 1, 1), (906, 1, 906, 906), device='cpu', dtype=torch.float32)
    buf359 = empty_strided((1, 906, 1, 1), (906, 1, 906, 906), device='cpu', dtype=torch.float32)
    buf361 = empty((906, ), device='cpu', dtype=torch.float32)
    buf362 = empty_strided((8, 906, 7, 7), (44394, 1, 6342, 906), device='cpu', dtype=torch.float32)
    buf363 = empty_strided((8, 906, 1, 1), (906, 1, 7248, 7248), device='cpu', dtype=torch.float32)
    buf364 = reinterpret_tensor(buf363, (8, 906, 1, 1), (906, 1, 906, 906), 0); del buf363  # reuse
    cpp_fused__native_batch_norm_legit_functional_mean_61(c_void_p(buf364.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(primals_82.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf362.data_ptr()))
    del primals_82
    # Source Nodes: [x_se_41], Original ATen: [aten.convolution]
    buf365 = extern_kernels.convolution(buf364, primals_200, primals_201, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf365, (8, 75, 1, 1), (75, 1, 75, 75))
    del primals_201
    buf366 = empty_strided((1, 75, 1, 1), (75, 1, 75, 75), device='cpu', dtype=torch.float32)
    buf367 = empty_strided((1, 75, 1, 1), (75, 1, 75, 75), device='cpu', dtype=torch.float32)
    buf369 = empty_strided((8, 75, 1, 1), (75, 1, 75, 75), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_62(c_void_p(buf365.data_ptr()), c_void_p(primals_202.data_ptr()), c_void_p(primals_203.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf369.data_ptr()))
    del primals_203
    # Source Nodes: [x_se_43], Original ATen: [aten.convolution]
    buf370 = extern_kernels.convolution(buf369, primals_204, primals_205, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf370, (8, 906, 1, 1), (906, 1, 906, 906))
    del primals_205
    buf371 = empty_strided((8, 906, 7, 7), (44394, 1, 6342, 906), device='cpu', dtype=torch.float32)
    cpp_fused_hardtanh_mul_sigmoid_63(c_void_p(buf362.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf371.data_ptr()))
    # Source Nodes: [x_279], Original ATen: [aten.convolution]
    buf372 = extern_kernels.convolution(buf371, primals_206, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf372, (8, 162, 7, 7), (7938, 1, 1134, 162))
    buf373 = empty_strided((1, 162, 1, 1), (162, 1, 162, 162), device='cpu', dtype=torch.float32)
    buf374 = empty_strided((1, 162, 1, 1), (162, 1, 162, 162), device='cpu', dtype=torch.float32)
    buf376 = empty((162, ), device='cpu', dtype=torch.float32)
    buf377 = empty_strided((8, 162, 7, 7), (7938, 1, 1134, 162), device='cpu', dtype=torch.float32)
    buf378 = buf377; del buf377  # reuse
    cpp_fused__native_batch_norm_legit_functional_cat_64(c_void_p(buf378.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(primals_84.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf376.data_ptr()))
    del primals_84
    # Source Nodes: [x_286], Original ATen: [aten.convolution]
    buf379 = extern_kernels.convolution(buf378, primals_207, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf379, (8, 972, 7, 7), (47628, 1, 6804, 972))
    buf380 = empty_strided((1, 972, 1, 1), (972, 1, 972, 972), device='cpu', dtype=torch.float32)
    buf381 = empty_strided((1, 972, 1, 1), (972, 1, 972, 972), device='cpu', dtype=torch.float32)
    buf383 = empty((972, ), device='cpu', dtype=torch.float32)
    buf384 = empty_strided((8, 972, 7, 7), (47628, 1, 6804, 972), device='cpu', dtype=torch.float32)
    buf385 = empty_strided((8, 972, 7, 7), (47628, 1, 6804, 972), device='cpu', dtype=torch.float32)
    buf448 = empty_strided((8, 972, 7, 7), (47628, 1, 6804, 972), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_65(c_void_p(buf379.data_ptr()), c_void_p(primals_85.data_ptr()), c_void_p(primals_86.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf448.data_ptr()))
    del primals_86
    # Source Nodes: [x_292], Original ATen: [aten.convolution]
    buf386 = extern_kernels.convolution(buf385, primals_208, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=972, bias=None)
    assert_size_stride(buf386, (8, 972, 7, 7), (47628, 1, 6804, 972))
    buf387 = empty_strided((1, 972, 1, 1), (972, 1, 972, 972), device='cpu', dtype=torch.float32)
    buf388 = empty_strided((1, 972, 1, 1), (972, 1, 972, 972), device='cpu', dtype=torch.float32)
    buf390 = empty((972, ), device='cpu', dtype=torch.float32)
    buf391 = buf384; del buf384  # reuse
    buf392 = empty_strided((8, 972, 1, 1), (972, 1, 7776, 7776), device='cpu', dtype=torch.float32)
    buf393 = reinterpret_tensor(buf392, (8, 972, 1, 1), (972, 1, 972, 972), 0); del buf392  # reuse
    cpp_fused__native_batch_norm_legit_functional_mean_66(c_void_p(buf393.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(primals_88.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(buf391.data_ptr()))
    del primals_88
    # Source Nodes: [x_se_45], Original ATen: [aten.convolution]
    buf394 = extern_kernels.convolution(buf393, primals_209, primals_210, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf394, (8, 81, 1, 1), (81, 1, 81, 81))
    del primals_210
    buf395 = empty_strided((1, 81, 1, 1), (81, 1, 81, 81), device='cpu', dtype=torch.float32)
    buf396 = empty_strided((1, 81, 1, 1), (81, 1, 81, 81), device='cpu', dtype=torch.float32)
    buf398 = empty_strided((8, 81, 1, 1), (81, 1, 81, 81), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_67(c_void_p(buf394.data_ptr()), c_void_p(primals_211.data_ptr()), c_void_p(primals_212.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf398.data_ptr()))
    del primals_212
    # Source Nodes: [x_se_47], Original ATen: [aten.convolution]
    buf399 = extern_kernels.convolution(buf398, primals_213, primals_214, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf399, (8, 972, 1, 1), (972, 1, 972, 972))
    del primals_214
    buf400 = empty_strided((8, 972, 7, 7), (47628, 1, 6804, 972), device='cpu', dtype=torch.float32)
    cpp_fused_hardtanh_mul_sigmoid_68(c_void_p(buf391.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf400.data_ptr()))
    # Source Nodes: [x_300], Original ATen: [aten.convolution]
    buf401 = extern_kernels.convolution(buf400, primals_215, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf401, (8, 174, 7, 7), (8526, 1, 1218, 174))
    buf402 = empty_strided((1, 174, 1, 1), (174, 1, 174, 174), device='cpu', dtype=torch.float32)
    buf403 = empty_strided((1, 174, 1, 1), (174, 1, 174, 174), device='cpu', dtype=torch.float32)
    buf405 = empty((174, ), device='cpu', dtype=torch.float32)
    buf406 = empty_strided((8, 174, 7, 7), (8526, 1, 1218, 174), device='cpu', dtype=torch.float32)
    buf407 = buf406; del buf406  # reuse
    cpp_fused__native_batch_norm_legit_functional_cat_69(c_void_p(buf407.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(primals_90.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf405.data_ptr()))
    del primals_90
    # Source Nodes: [x_307], Original ATen: [aten.convolution]
    buf408 = extern_kernels.convolution(buf407, primals_216, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf408, (8, 1044, 7, 7), (51156, 1, 7308, 1044))
    buf409 = empty_strided((1, 1044, 1, 1), (1044, 1, 1044, 1044), device='cpu', dtype=torch.float32)
    buf410 = empty_strided((1, 1044, 1, 1), (1044, 1, 1044, 1044), device='cpu', dtype=torch.float32)
    buf412 = empty((1044, ), device='cpu', dtype=torch.float32)
    buf413 = empty_strided((8, 1044, 7, 7), (51156, 1, 7308, 1044), device='cpu', dtype=torch.float32)
    buf414 = empty_strided((8, 1044, 7, 7), (51156, 1, 7308, 1044), device='cpu', dtype=torch.float32)
    buf447 = empty_strided((8, 1044, 7, 7), (51156, 1, 7308, 1044), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_70(c_void_p(buf408.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(primals_92.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf447.data_ptr()))
    del primals_92
    # Source Nodes: [x_313], Original ATen: [aten.convolution]
    buf415 = extern_kernels.convolution(buf414, primals_217, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1044, bias=None)
    assert_size_stride(buf415, (8, 1044, 7, 7), (51156, 1, 7308, 1044))
    buf416 = empty_strided((1, 1044, 1, 1), (1044, 1, 1044, 1044), device='cpu', dtype=torch.float32)
    buf417 = empty_strided((1, 1044, 1, 1), (1044, 1, 1044, 1044), device='cpu', dtype=torch.float32)
    buf419 = empty((1044, ), device='cpu', dtype=torch.float32)
    buf420 = buf413; del buf413  # reuse
    buf421 = empty_strided((8, 1044, 1, 1), (1044, 1, 8352, 8352), device='cpu', dtype=torch.float32)
    buf422 = reinterpret_tensor(buf421, (8, 1044, 1, 1), (1044, 1, 1044, 1044), 0); del buf421  # reuse
    cpp_fused__native_batch_norm_legit_functional_mean_71(c_void_p(buf422.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(primals_94.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf420.data_ptr()))
    del primals_94
    # Source Nodes: [x_se_49], Original ATen: [aten.convolution]
    buf423 = extern_kernels.convolution(buf422, primals_218, primals_219, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf423, (8, 87, 1, 1), (87, 1, 87, 87))
    del primals_219
    buf424 = empty_strided((1, 87, 1, 1), (87, 1, 87, 87), device='cpu', dtype=torch.float32)
    buf425 = empty_strided((1, 87, 1, 1), (87, 1, 87, 87), device='cpu', dtype=torch.float32)
    buf427 = empty_strided((8, 87, 1, 1), (87, 1, 87, 87), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_72(c_void_p(buf423.data_ptr()), c_void_p(primals_220.data_ptr()), c_void_p(primals_221.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf427.data_ptr()))
    del primals_221
    # Source Nodes: [x_se_51], Original ATen: [aten.convolution]
    buf428 = extern_kernels.convolution(buf427, primals_222, primals_223, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf428, (8, 1044, 1, 1), (1044, 1, 1044, 1044))
    del primals_223
    buf429 = empty_strided((8, 1044, 7, 7), (51156, 1, 7308, 1044), device='cpu', dtype=torch.float32)
    cpp_fused_hardtanh_mul_sigmoid_73(c_void_p(buf420.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(buf429.data_ptr()))
    # Source Nodes: [x_321], Original ATen: [aten.convolution]
    buf430 = extern_kernels.convolution(buf429, primals_224, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf430, (8, 185, 7, 7), (9065, 1, 1295, 185))
    buf431 = empty_strided((1, 185, 1, 1), (185, 1, 185, 185), device='cpu', dtype=torch.float32)
    buf432 = empty_strided((1, 185, 1, 1), (185, 1, 185, 185), device='cpu', dtype=torch.float32)
    buf434 = empty((185, ), device='cpu', dtype=torch.float32)
    buf435 = empty_strided((8, 185, 7, 7), (9065, 1, 1295, 185), device='cpu', dtype=torch.float32)
    buf436 = buf435; del buf435  # reuse
    cpp_fused__native_batch_norm_legit_functional_cat_74(c_void_p(buf436.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(primals_96.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(buf434.data_ptr()))
    del primals_96
    # Source Nodes: [x_328], Original ATen: [aten.convolution]
    buf437 = extern_kernels.convolution(buf436, primals_225, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf437, (8, 1280, 7, 7), (62720, 1, 8960, 1280))
    buf438 = empty_strided((1, 1280, 1, 1), (1280, 1, 1280, 1280), device='cpu', dtype=torch.float32)
    buf439 = empty_strided((1, 1280, 1, 1), (1280, 1, 1280, 1280), device='cpu', dtype=torch.float32)
    buf441 = empty((1280, ), device='cpu', dtype=torch.float32)
    buf442 = empty_strided((8, 1280, 7, 7), (62720, 1, 8960, 1280), device='cpu', dtype=torch.float32)
    buf443 = empty_strided((8, 1280, 1, 1), (1280, 1, 10240, 10240), device='cpu', dtype=torch.float32)
    buf444 = reinterpret_tensor(buf443, (8, 1280), (1280, 1), 0); del buf443  # reuse
    cpp_fused__native_batch_norm_legit_functional_mean_silu_view_75(c_void_p(buf444.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(primals_97.data_ptr()), c_void_p(primals_98.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(buf442.data_ptr()))
    del primals_98
    buf445 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_339], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_227, buf444, reinterpret_tensor(primals_226, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf445)
    del primals_227
    buf446 = buf442; del buf442  # reuse
    buf449 = buf355; del buf355  # reuse
    buf450 = buf326; del buf326  # reuse
    buf451 = buf298; del buf298  # reuse
    buf452 = buf269; del buf269  # reuse
    buf453 = buf240; del buf240  # reuse
    buf454 = buf211; del buf211  # reuse
    buf455 = buf182; del buf182  # reuse
    buf456 = buf153; del buf153  # reuse
    buf457 = buf125; del buf125  # reuse
    buf458 = buf96; del buf96  # reuse
    buf459 = buf68; del buf68  # reuse
    buf460 = empty_strided((8, 162, 56, 56), (508032, 1, 9072, 162), device='cpu', dtype=torch.bool)
    buf461 = buf47; del buf47  # reuse
    buf462 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cpu', dtype=torch.bool)
    buf463 = buf27; del buf27  # reuse
    buf464 = empty_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.bool)
    buf465 = buf7; del buf7  # reuse
    buf472 = reinterpret_tensor(buf4, (32, ), (1, ), 0); del buf4  # reuse
    buf480 = reinterpret_tensor(buf11, (32, ), (1, ), 0); del buf11  # reuse
    buf488 = reinterpret_tensor(buf18, (16, ), (1, ), 0); del buf18  # reuse
    buf496 = reinterpret_tensor(buf24, (96, ), (1, ), 0); del buf24  # reuse
    buf504 = reinterpret_tensor(buf31, (96, ), (1, ), 0); del buf31  # reuse
    buf512 = reinterpret_tensor(buf38, (27, ), (1, ), 0); del buf38  # reuse
    buf520 = reinterpret_tensor(buf44, (162, ), (1, ), 0); del buf44  # reuse
    buf528 = reinterpret_tensor(buf51, (162, ), (1, ), 0); del buf51  # reuse
    buf536 = reinterpret_tensor(buf58, (38, ), (1, ), 0); del buf58  # reuse
    buf544 = reinterpret_tensor(buf65, (228, ), (1, ), 0); del buf65  # reuse
    buf552 = reinterpret_tensor(buf72, (228, ), (1, ), 0); del buf72  # reuse
    buf560 = reinterpret_tensor(buf87, (50, ), (1, ), 0); del buf87  # reuse
    buf568 = reinterpret_tensor(buf93, (300, ), (1, ), 0); del buf93  # reuse
    buf576 = reinterpret_tensor(buf100, (300, ), (1, ), 0); del buf100  # reuse
    buf584 = reinterpret_tensor(buf115, (61, ), (1, ), 0); del buf115  # reuse
    buf592 = reinterpret_tensor(buf122, (366, ), (1, ), 0); del buf122  # reuse
    buf600 = reinterpret_tensor(buf129, (366, ), (1, ), 0); del buf129  # reuse
    buf608 = reinterpret_tensor(buf144, (72, ), (1, ), 0); del buf144  # reuse
    buf616 = reinterpret_tensor(buf150, (432, ), (1, ), 0); del buf150  # reuse
    buf624 = reinterpret_tensor(buf157, (432, ), (1, ), 0); del buf157  # reuse
    buf632 = reinterpret_tensor(buf172, (84, ), (1, ), 0); del buf172  # reuse
    buf640 = reinterpret_tensor(buf179, (504, ), (1, ), 0); del buf179  # reuse
    buf648 = reinterpret_tensor(buf186, (504, ), (1, ), 0); del buf186  # reuse
    buf656 = reinterpret_tensor(buf201, (95, ), (1, ), 0); del buf201  # reuse
    buf664 = reinterpret_tensor(buf208, (570, ), (1, ), 0); del buf208  # reuse
    buf672 = reinterpret_tensor(buf215, (570, ), (1, ), 0); del buf215  # reuse
    buf680 = reinterpret_tensor(buf230, (106, ), (1, ), 0); del buf230  # reuse
    buf688 = reinterpret_tensor(buf237, (636, ), (1, ), 0); del buf237  # reuse
    buf696 = reinterpret_tensor(buf244, (636, ), (1, ), 0); del buf244  # reuse
    buf704 = reinterpret_tensor(buf259, (117, ), (1, ), 0); del buf259  # reuse
    buf712 = reinterpret_tensor(buf266, (702, ), (1, ), 0); del buf266  # reuse
    buf720 = reinterpret_tensor(buf273, (702, ), (1, ), 0); del buf273  # reuse
    buf728 = reinterpret_tensor(buf288, (128, ), (1, ), 0); del buf288  # reuse
    buf736 = reinterpret_tensor(buf295, (768, ), (1, ), 0); del buf295  # reuse
    buf744 = reinterpret_tensor(buf302, (768, ), (1, ), 0); del buf302  # reuse
    buf752 = reinterpret_tensor(buf317, (140, ), (1, ), 0); del buf317  # reuse
    buf760 = reinterpret_tensor(buf323, (840, ), (1, ), 0); del buf323  # reuse
    buf768 = reinterpret_tensor(buf330, (840, ), (1, ), 0); del buf330  # reuse
    buf776 = reinterpret_tensor(buf345, (151, ), (1, ), 0); del buf345  # reuse
    buf784 = reinterpret_tensor(buf352, (906, ), (1, ), 0); del buf352  # reuse
    buf792 = reinterpret_tensor(buf359, (906, ), (1, ), 0); del buf359  # reuse
    buf800 = reinterpret_tensor(buf374, (162, ), (1, ), 0); del buf374  # reuse
    buf808 = reinterpret_tensor(buf381, (972, ), (1, ), 0); del buf381  # reuse
    buf816 = reinterpret_tensor(buf388, (972, ), (1, ), 0); del buf388  # reuse
    buf824 = reinterpret_tensor(buf403, (174, ), (1, ), 0); del buf403  # reuse
    buf832 = reinterpret_tensor(buf410, (1044, ), (1, ), 0); del buf410  # reuse
    buf840 = reinterpret_tensor(buf417, (1044, ), (1, ), 0); del buf417  # reuse
    buf848 = reinterpret_tensor(buf432, (185, ), (1, ), 0); del buf432  # reuse
    buf856 = reinterpret_tensor(buf439, (1280, ), (1, ), 0); del buf439  # reuse
    buf859 = reinterpret_tensor(buf79, (19, ), (1, ), 0); del buf79  # reuse
    buf862 = reinterpret_tensor(buf80, (19, ), (1, ), 0); del buf80  # reuse
    buf867 = reinterpret_tensor(buf107, (25, ), (1, ), 0); del buf107  # reuse
    buf870 = reinterpret_tensor(buf108, (25, ), (1, ), 0); del buf108  # reuse
    buf875 = reinterpret_tensor(buf136, (30, ), (1, ), 0); del buf136  # reuse
    buf878 = reinterpret_tensor(buf137, (30, ), (1, ), 0); del buf137  # reuse
    buf883 = reinterpret_tensor(buf164, (36, ), (1, ), 0); del buf164  # reuse
    buf886 = reinterpret_tensor(buf165, (36, ), (1, ), 0); del buf165  # reuse
    buf891 = reinterpret_tensor(buf193, (42, ), (1, ), 0); del buf193  # reuse
    buf894 = reinterpret_tensor(buf194, (42, ), (1, ), 0); del buf194  # reuse
    buf899 = reinterpret_tensor(buf222, (47, ), (1, ), 0); del buf222  # reuse
    buf902 = reinterpret_tensor(buf223, (47, ), (1, ), 0); del buf223  # reuse
    buf907 = reinterpret_tensor(buf251, (53, ), (1, ), 0); del buf251  # reuse
    buf910 = reinterpret_tensor(buf252, (53, ), (1, ), 0); del buf252  # reuse
    buf915 = reinterpret_tensor(buf280, (58, ), (1, ), 0); del buf280  # reuse
    buf918 = reinterpret_tensor(buf281, (58, ), (1, ), 0); del buf281  # reuse
    buf923 = reinterpret_tensor(buf309, (64, ), (1, ), 0); del buf309  # reuse
    buf926 = reinterpret_tensor(buf310, (64, ), (1, ), 0); del buf310  # reuse
    buf931 = reinterpret_tensor(buf337, (70, ), (1, ), 0); del buf337  # reuse
    buf934 = reinterpret_tensor(buf338, (70, ), (1, ), 0); del buf338  # reuse
    buf939 = reinterpret_tensor(buf366, (75, ), (1, ), 0); del buf366  # reuse
    buf942 = reinterpret_tensor(buf367, (75, ), (1, ), 0); del buf367  # reuse
    cpp_fused__native_batch_norm_legit_functional_add_fill_hardtanh_backward_mul_sigmoid_sub_76(c_void_p(buf446.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf453.data_ptr()), c_void_p(buf454.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(buf463.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(buf472.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(buf488.data_ptr()), c_void_p(buf496.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(buf512.data_ptr()), c_void_p(buf520.data_ptr()), c_void_p(buf528.data_ptr()), c_void_p(buf536.data_ptr()), c_void_p(buf544.data_ptr()), c_void_p(buf552.data_ptr()), c_void_p(buf560.data_ptr()), c_void_p(buf568.data_ptr()), c_void_p(buf576.data_ptr()), c_void_p(buf584.data_ptr()), c_void_p(buf592.data_ptr()), c_void_p(buf600.data_ptr()), c_void_p(buf608.data_ptr()), c_void_p(buf616.data_ptr()), c_void_p(buf624.data_ptr()), c_void_p(buf632.data_ptr()), c_void_p(buf640.data_ptr()), c_void_p(buf648.data_ptr()), c_void_p(buf656.data_ptr()), c_void_p(buf664.data_ptr()), c_void_p(buf672.data_ptr()), c_void_p(buf680.data_ptr()), c_void_p(buf688.data_ptr()), c_void_p(buf696.data_ptr()), c_void_p(buf704.data_ptr()), c_void_p(buf712.data_ptr()), c_void_p(buf720.data_ptr()), c_void_p(buf728.data_ptr()), c_void_p(buf736.data_ptr()), c_void_p(buf744.data_ptr()), c_void_p(buf752.data_ptr()), c_void_p(buf760.data_ptr()), c_void_p(buf768.data_ptr()), c_void_p(buf776.data_ptr()), c_void_p(buf784.data_ptr()), c_void_p(buf792.data_ptr()), c_void_p(buf800.data_ptr()), c_void_p(buf808.data_ptr()), c_void_p(buf816.data_ptr()), c_void_p(buf824.data_ptr()), c_void_p(buf832.data_ptr()), c_void_p(buf840.data_ptr()), c_void_p(buf848.data_ptr()), c_void_p(buf856.data_ptr()), c_void_p(buf859.data_ptr()), c_void_p(buf862.data_ptr()), c_void_p(buf867.data_ptr()), c_void_p(buf870.data_ptr()), c_void_p(buf875.data_ptr()), c_void_p(buf878.data_ptr()), c_void_p(buf883.data_ptr()), c_void_p(buf886.data_ptr()), c_void_p(buf891.data_ptr()), c_void_p(buf894.data_ptr()), c_void_p(buf899.data_ptr()), c_void_p(buf902.data_ptr()), c_void_p(buf907.data_ptr()), c_void_p(buf910.data_ptr()), c_void_p(buf915.data_ptr()), c_void_p(buf918.data_ptr()), c_void_p(buf923.data_ptr()), c_void_p(buf926.data_ptr()), c_void_p(buf931.data_ptr()), c_void_p(buf934.data_ptr()), c_void_p(buf939.data_ptr()), c_void_p(buf942.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(primals_228.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(primals_229.data_ptr()), c_void_p(primals_230.data_ptr()), c_void_p(primals_231.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(primals_232.data_ptr()), c_void_p(primals_233.data_ptr()), c_void_p(primals_234.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(primals_235.data_ptr()), c_void_p(primals_236.data_ptr()), c_void_p(primals_237.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(primals_238.data_ptr()), c_void_p(primals_239.data_ptr()), c_void_p(primals_240.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(primals_241.data_ptr()), c_void_p(primals_242.data_ptr()), c_void_p(primals_243.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(primals_244.data_ptr()), c_void_p(primals_245.data_ptr()), c_void_p(primals_246.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(primals_247.data_ptr()), c_void_p(primals_248.data_ptr()), c_void_p(primals_249.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(primals_250.data_ptr()), c_void_p(primals_251.data_ptr()), c_void_p(primals_252.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(primals_253.data_ptr()), c_void_p(primals_254.data_ptr()), c_void_p(primals_255.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(primals_256.data_ptr()), c_void_p(primals_257.data_ptr()), c_void_p(primals_258.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(primals_259.data_ptr()), c_void_p(primals_260.data_ptr()), c_void_p(primals_261.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(primals_262.data_ptr()), c_void_p(primals_263.data_ptr()), c_void_p(primals_264.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(primals_265.data_ptr()), c_void_p(primals_266.data_ptr()), c_void_p(primals_267.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(primals_268.data_ptr()), c_void_p(primals_269.data_ptr()), c_void_p(primals_270.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(primals_271.data_ptr()), c_void_p(primals_272.data_ptr()), c_void_p(primals_273.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(primals_274.data_ptr()), c_void_p(primals_275.data_ptr()), c_void_p(primals_276.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(primals_277.data_ptr()), c_void_p(primals_278.data_ptr()), c_void_p(primals_279.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(primals_280.data_ptr()), c_void_p(primals_281.data_ptr()), c_void_p(primals_282.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(primals_283.data_ptr()), c_void_p(primals_284.data_ptr()), c_void_p(primals_285.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(primals_286.data_ptr()), c_void_p(primals_287.data_ptr()), c_void_p(primals_288.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(primals_289.data_ptr()), c_void_p(primals_290.data_ptr()), c_void_p(primals_291.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(primals_292.data_ptr()), c_void_p(primals_293.data_ptr()), c_void_p(primals_294.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(primals_295.data_ptr()), c_void_p(primals_296.data_ptr()), c_void_p(primals_297.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(primals_298.data_ptr()), c_void_p(primals_299.data_ptr()), c_void_p(primals_300.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(primals_301.data_ptr()), c_void_p(primals_302.data_ptr()), c_void_p(primals_303.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(primals_304.data_ptr()), c_void_p(primals_305.data_ptr()), c_void_p(primals_306.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(primals_307.data_ptr()), c_void_p(primals_308.data_ptr()), c_void_p(primals_309.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(primals_310.data_ptr()), c_void_p(primals_311.data_ptr()), c_void_p(primals_312.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(primals_313.data_ptr()), c_void_p(primals_314.data_ptr()), c_void_p(primals_315.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(primals_316.data_ptr()), c_void_p(primals_317.data_ptr()), c_void_p(primals_318.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(primals_319.data_ptr()), c_void_p(primals_320.data_ptr()), c_void_p(primals_321.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(primals_322.data_ptr()), c_void_p(primals_323.data_ptr()), c_void_p(primals_324.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(primals_325.data_ptr()), c_void_p(primals_326.data_ptr()), c_void_p(primals_327.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(primals_328.data_ptr()), c_void_p(primals_329.data_ptr()), c_void_p(primals_330.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(primals_331.data_ptr()), c_void_p(primals_332.data_ptr()), c_void_p(primals_333.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(primals_334.data_ptr()), c_void_p(primals_335.data_ptr()), c_void_p(primals_336.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(primals_337.data_ptr()), c_void_p(primals_338.data_ptr()), c_void_p(primals_339.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(primals_340.data_ptr()), c_void_p(primals_341.data_ptr()), c_void_p(primals_342.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(primals_343.data_ptr()), c_void_p(primals_344.data_ptr()), c_void_p(primals_345.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(primals_346.data_ptr()), c_void_p(primals_347.data_ptr()), c_void_p(primals_348.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(primals_349.data_ptr()), c_void_p(primals_350.data_ptr()), c_void_p(primals_351.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(primals_352.data_ptr()), c_void_p(primals_353.data_ptr()), c_void_p(primals_354.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(primals_355.data_ptr()), c_void_p(primals_356.data_ptr()), c_void_p(primals_357.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(primals_358.data_ptr()), c_void_p(primals_359.data_ptr()), c_void_p(primals_360.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(primals_361.data_ptr()), c_void_p(primals_362.data_ptr()), c_void_p(primals_363.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(primals_364.data_ptr()), c_void_p(primals_365.data_ptr()), c_void_p(primals_366.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(primals_367.data_ptr()), c_void_p(primals_368.data_ptr()), c_void_p(primals_369.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(primals_370.data_ptr()), c_void_p(primals_371.data_ptr()), c_void_p(primals_372.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(primals_373.data_ptr()), c_void_p(primals_374.data_ptr()), c_void_p(primals_375.data_ptr()), c_void_p(primals_376.data_ptr()), c_void_p(primals_377.data_ptr()), c_void_p(primals_378.data_ptr()), c_void_p(primals_379.data_ptr()), c_void_p(primals_380.data_ptr()), c_void_p(primals_381.data_ptr()), c_void_p(primals_382.data_ptr()), c_void_p(primals_383.data_ptr()), c_void_p(primals_384.data_ptr()), c_void_p(primals_385.data_ptr()), c_void_p(primals_386.data_ptr()), c_void_p(primals_387.data_ptr()), c_void_p(primals_388.data_ptr()), c_void_p(primals_389.data_ptr()), c_void_p(primals_390.data_ptr()), c_void_p(primals_391.data_ptr()), c_void_p(primals_392.data_ptr()), c_void_p(primals_393.data_ptr()), c_void_p(primals_394.data_ptr()), c_void_p(primals_395.data_ptr()), c_void_p(primals_396.data_ptr()), c_void_p(primals_397.data_ptr()), c_void_p(primals_398.data_ptr()), c_void_p(primals_399.data_ptr()), c_void_p(primals_400.data_ptr()), c_void_p(primals_401.data_ptr()), c_void_p(primals_402.data_ptr()), c_void_p(primals_403.data_ptr()), c_void_p(primals_404.data_ptr()), c_void_p(primals_405.data_ptr()), c_void_p(primals_406.data_ptr()), c_void_p(primals_407.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(primals_228.data_ptr()), c_void_p(primals_229.data_ptr()), c_void_p(primals_230.data_ptr()), c_void_p(primals_231.data_ptr()), c_void_p(primals_232.data_ptr()), c_void_p(primals_233.data_ptr()), c_void_p(primals_234.data_ptr()), c_void_p(primals_235.data_ptr()), c_void_p(primals_236.data_ptr()), c_void_p(primals_237.data_ptr()), c_void_p(primals_238.data_ptr()), c_void_p(primals_239.data_ptr()), c_void_p(primals_240.data_ptr()), c_void_p(primals_241.data_ptr()), c_void_p(primals_242.data_ptr()), c_void_p(primals_243.data_ptr()), c_void_p(primals_244.data_ptr()), c_void_p(primals_245.data_ptr()), c_void_p(primals_246.data_ptr()), c_void_p(primals_247.data_ptr()), c_void_p(primals_248.data_ptr()), c_void_p(primals_249.data_ptr()), c_void_p(primals_250.data_ptr()), c_void_p(primals_251.data_ptr()), c_void_p(primals_252.data_ptr()), c_void_p(primals_253.data_ptr()), c_void_p(primals_254.data_ptr()), c_void_p(primals_255.data_ptr()), c_void_p(primals_256.data_ptr()), c_void_p(primals_257.data_ptr()), c_void_p(primals_258.data_ptr()), c_void_p(primals_259.data_ptr()), c_void_p(primals_260.data_ptr()), c_void_p(primals_261.data_ptr()), c_void_p(primals_262.data_ptr()), c_void_p(primals_263.data_ptr()), c_void_p(primals_264.data_ptr()), c_void_p(primals_265.data_ptr()), c_void_p(primals_266.data_ptr()), c_void_p(primals_267.data_ptr()), c_void_p(primals_268.data_ptr()), c_void_p(primals_269.data_ptr()), c_void_p(primals_270.data_ptr()), c_void_p(primals_271.data_ptr()), c_void_p(primals_272.data_ptr()), c_void_p(primals_273.data_ptr()), c_void_p(primals_274.data_ptr()), c_void_p(primals_275.data_ptr()), c_void_p(primals_276.data_ptr()), c_void_p(primals_277.data_ptr()), c_void_p(primals_278.data_ptr()), c_void_p(primals_279.data_ptr()), c_void_p(primals_280.data_ptr()), c_void_p(primals_281.data_ptr()), c_void_p(primals_282.data_ptr()), c_void_p(primals_283.data_ptr()), c_void_p(primals_284.data_ptr()), c_void_p(primals_285.data_ptr()), c_void_p(primals_286.data_ptr()), c_void_p(primals_287.data_ptr()), c_void_p(primals_288.data_ptr()), c_void_p(primals_289.data_ptr()), c_void_p(primals_290.data_ptr()), c_void_p(primals_291.data_ptr()), c_void_p(primals_292.data_ptr()), c_void_p(primals_293.data_ptr()), c_void_p(primals_294.data_ptr()), c_void_p(primals_295.data_ptr()), c_void_p(primals_296.data_ptr()), c_void_p(primals_297.data_ptr()), c_void_p(primals_298.data_ptr()), c_void_p(primals_299.data_ptr()), c_void_p(primals_300.data_ptr()), c_void_p(primals_301.data_ptr()), c_void_p(primals_302.data_ptr()), c_void_p(primals_303.data_ptr()), c_void_p(primals_304.data_ptr()), c_void_p(primals_305.data_ptr()), c_void_p(primals_306.data_ptr()), c_void_p(primals_307.data_ptr()), c_void_p(primals_308.data_ptr()), c_void_p(primals_309.data_ptr()), c_void_p(primals_310.data_ptr()), c_void_p(primals_311.data_ptr()), c_void_p(primals_312.data_ptr()), c_void_p(primals_313.data_ptr()), c_void_p(primals_314.data_ptr()), c_void_p(primals_315.data_ptr()), c_void_p(primals_316.data_ptr()), c_void_p(primals_317.data_ptr()), c_void_p(primals_318.data_ptr()), c_void_p(primals_319.data_ptr()), c_void_p(primals_320.data_ptr()), c_void_p(primals_321.data_ptr()), c_void_p(primals_322.data_ptr()), c_void_p(primals_323.data_ptr()), c_void_p(primals_324.data_ptr()), c_void_p(primals_325.data_ptr()), c_void_p(primals_326.data_ptr()), c_void_p(primals_327.data_ptr()), c_void_p(primals_328.data_ptr()), c_void_p(primals_329.data_ptr()), c_void_p(primals_330.data_ptr()), c_void_p(primals_331.data_ptr()), c_void_p(primals_332.data_ptr()), c_void_p(primals_333.data_ptr()), c_void_p(primals_334.data_ptr()), c_void_p(primals_335.data_ptr()), c_void_p(primals_336.data_ptr()), c_void_p(primals_337.data_ptr()), c_void_p(primals_338.data_ptr()), c_void_p(primals_339.data_ptr()), c_void_p(primals_340.data_ptr()), c_void_p(primals_341.data_ptr()), c_void_p(primals_342.data_ptr()), c_void_p(primals_343.data_ptr()), c_void_p(primals_344.data_ptr()), c_void_p(primals_345.data_ptr()), c_void_p(primals_346.data_ptr()), c_void_p(primals_347.data_ptr()), c_void_p(primals_348.data_ptr()), c_void_p(primals_349.data_ptr()), c_void_p(primals_350.data_ptr()), c_void_p(primals_351.data_ptr()), c_void_p(primals_352.data_ptr()), c_void_p(primals_353.data_ptr()), c_void_p(primals_354.data_ptr()), c_void_p(primals_355.data_ptr()), c_void_p(primals_356.data_ptr()), c_void_p(primals_357.data_ptr()), c_void_p(primals_358.data_ptr()), c_void_p(primals_359.data_ptr()), c_void_p(primals_360.data_ptr()), c_void_p(primals_361.data_ptr()), c_void_p(primals_362.data_ptr()), c_void_p(primals_363.data_ptr()), c_void_p(primals_364.data_ptr()), c_void_p(primals_365.data_ptr()), c_void_p(primals_366.data_ptr()), c_void_p(primals_367.data_ptr()), c_void_p(primals_368.data_ptr()), c_void_p(primals_369.data_ptr()), c_void_p(primals_370.data_ptr()), c_void_p(primals_371.data_ptr()), c_void_p(primals_372.data_ptr()), c_void_p(primals_373.data_ptr()), c_void_p(primals_374.data_ptr()), c_void_p(primals_375.data_ptr()), c_void_p(primals_376.data_ptr()), c_void_p(primals_377.data_ptr()), c_void_p(primals_378.data_ptr()), c_void_p(primals_379.data_ptr()), c_void_p(primals_380.data_ptr()), c_void_p(primals_381.data_ptr()), c_void_p(primals_382.data_ptr()), c_void_p(primals_383.data_ptr()), c_void_p(primals_384.data_ptr()), c_void_p(primals_385.data_ptr()), c_void_p(primals_386.data_ptr()), c_void_p(primals_387.data_ptr()), c_void_p(primals_388.data_ptr()), c_void_p(primals_389.data_ptr()), c_void_p(primals_390.data_ptr()), c_void_p(primals_391.data_ptr()), c_void_p(primals_392.data_ptr()), c_void_p(primals_393.data_ptr()), c_void_p(primals_394.data_ptr()), c_void_p(primals_395.data_ptr()), c_void_p(primals_396.data_ptr()), c_void_p(primals_397.data_ptr()), c_void_p(primals_398.data_ptr()), c_void_p(primals_399.data_ptr()), c_void_p(primals_400.data_ptr()), c_void_p(primals_401.data_ptr()), c_void_p(primals_402.data_ptr()), c_void_p(primals_403.data_ptr()), c_void_p(primals_404.data_ptr()), c_void_p(primals_405.data_ptr()), c_void_p(primals_406.data_ptr()), c_void_p(primals_407.data_ptr()))
    del buf14
    del buf34
    del buf472
    del buf480
    del buf488
    del buf496
    del buf504
    del buf512
    del buf520
    del buf528
    del buf536
    del buf54
    del buf544
    del buf552
    del buf560
    del buf568
    del buf576
    del buf584
    del buf592
    del buf600
    del buf608
    del buf616
    del buf624
    del buf632
    del buf640
    del buf648
    del buf656
    del buf664
    del buf672
    del buf680
    del buf688
    del buf696
    del buf704
    del buf712
    del buf720
    del buf728
    del buf736
    del buf744
    del buf752
    del buf760
    del buf768
    del buf776
    del buf784
    del buf792
    del buf800
    del buf808
    del buf816
    del buf824
    del buf832
    del buf840
    del buf848
    del buf856
    del buf859
    del buf862
    del buf867
    del buf870
    del buf875
    del buf878
    del buf883
    del buf886
    del buf891
    del buf894
    del buf899
    del buf902
    del buf907
    del buf910
    del buf915
    del buf918
    del buf923
    del buf926
    del buf931
    del buf934
    del buf939
    del buf942
    del primals_228
    del primals_229
    del primals_230
    del primals_231
    del primals_232
    del primals_233
    del primals_234
    del primals_235
    del primals_236
    del primals_237
    del primals_238
    del primals_239
    del primals_240
    del primals_241
    del primals_242
    del primals_243
    del primals_244
    del primals_245
    del primals_246
    del primals_247
    del primals_248
    del primals_249
    del primals_250
    del primals_251
    del primals_252
    del primals_253
    del primals_254
    del primals_255
    del primals_256
    del primals_257
    del primals_258
    del primals_259
    del primals_260
    del primals_261
    del primals_262
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
    buf947 = reinterpret_tensor(buf395, (81, ), (1, ), 0); del buf395  # reuse
    buf950 = reinterpret_tensor(buf396, (81, ), (1, ), 0); del buf396  # reuse
    buf955 = reinterpret_tensor(buf424, (87, ), (1, ), 0); del buf424  # reuse
    buf958 = reinterpret_tensor(buf425, (87, ), (1, ), 0); del buf425  # reuse
    cpp_fused__native_batch_norm_legit_functional_add_77(c_void_p(buf947.data_ptr()), c_void_p(buf950.data_ptr()), c_void_p(buf955.data_ptr()), c_void_p(buf958.data_ptr()), c_void_p(primals_408.data_ptr()), c_void_p(primals_409.data_ptr()), c_void_p(primals_410.data_ptr()), c_void_p(primals_411.data_ptr()), c_void_p(primals_412.data_ptr()), c_void_p(primals_413.data_ptr()), c_void_p(primals_408.data_ptr()), c_void_p(primals_409.data_ptr()), c_void_p(primals_410.data_ptr()), c_void_p(primals_411.data_ptr()), c_void_p(primals_412.data_ptr()), c_void_p(primals_413.data_ptr()))
    del buf947
    del buf950
    del buf955
    del buf958
    del primals_408
    del primals_409
    del primals_410
    del primals_411
    del primals_412
    del primals_413
    return (buf445, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, buf0, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_112, primals_114, primals_116, primals_117, primals_118, primals_119, primals_121, primals_123, primals_125, primals_126, primals_127, primals_128, primals_130, primals_132, primals_134, primals_135, primals_136, primals_137, primals_139, primals_141, primals_143, primals_144, primals_145, primals_146, primals_148, primals_150, primals_152, primals_153, primals_154, primals_155, primals_157, primals_159, primals_161, primals_162, primals_163, primals_164, primals_166, primals_168, primals_170, primals_171, primals_172, primals_173, primals_175, primals_177, primals_179, primals_180, primals_181, primals_182, primals_184, primals_186, primals_188, primals_189, primals_190, primals_191, primals_193, primals_195, primals_197, primals_198, primals_199, primals_200, primals_202, primals_204, primals_206, primals_207, primals_208, primals_209, primals_211, primals_213, primals_215, primals_216, primals_217, primals_218, primals_220, primals_222, primals_224, primals_225, buf1, buf2, buf6, buf8, buf9, buf13, buf15, buf16, buf20, buf21, buf22, buf26, buf28, buf29, buf33, buf35, buf36, buf40, buf41, buf42, buf46, buf48, buf49, buf53, buf55, buf56, buf60, buf62, buf63, buf67, buf69, buf70, buf74, buf75, buf77, buf78, buf82, buf83, buf84, buf85, buf89, buf90, buf91, buf95, buf97, buf98, buf102, buf103, buf105, buf106, buf110, buf111, buf112, buf113, buf117, buf119, buf120, buf124, buf126, buf127, buf131, buf132, buf134, buf135, buf139, buf140, buf141, buf142, buf146, buf147, buf148, buf152, buf154, buf155, buf159, buf160, buf162, buf163, buf167, buf168, buf169, buf170, buf174, buf176, buf177, buf181, buf183, buf184, buf188, buf189, buf191, buf192, buf196, buf197, buf198, buf199, buf203, buf205, buf206, buf210, buf212, buf213, buf217, buf218, buf220, buf221, buf225, buf226, buf227, buf228, buf232, buf234, buf235, buf239, buf241, buf242, buf246, buf247, buf249, buf250, buf254, buf255, buf256, buf257, buf261, buf263, buf264, buf268, buf270, buf271, buf275, buf276, buf278, buf279, buf283, buf284, buf285, buf286, buf290, buf292, buf293, buf297, buf299, buf300, buf304, buf305, buf307, buf308, buf312, buf313, buf314, buf315, buf319, buf320, buf321, buf325, buf327, buf328, buf332, buf333, buf335, buf336, buf340, buf341, buf342, buf343, buf347, buf349, buf350, buf354, buf356, buf357, buf361, buf362, buf364, buf365, buf369, buf370, buf371, buf372, buf376, buf378, buf379, buf383, buf385, buf386, buf390, buf391, buf393, buf394, buf398, buf399, buf400, buf401, buf405, buf407, buf408, buf412, buf414, buf415, buf419, buf420, buf422, buf423, buf427, buf428, buf429, buf430, buf434, buf436, buf437, buf441, buf444, reinterpret_tensor(primals_226, (1000, 1280), (1280, 1), 0), buf446, reinterpret_tensor(buf438, (1, 1280, 1, 1), (1280, 1, 1, 1), 0), reinterpret_tensor(buf431, (1, 185, 1, 1), (185, 1, 1, 1), 0), reinterpret_tensor(buf416, (1, 1044, 1, 1), (1044, 1, 1, 1), 0), buf447, reinterpret_tensor(buf409, (1, 1044, 1, 1), (1044, 1, 1, 1), 0), reinterpret_tensor(buf402, (1, 174, 1, 1), (174, 1, 1, 1), 0), reinterpret_tensor(buf387, (1, 972, 1, 1), (972, 1, 1, 1), 0), buf448, reinterpret_tensor(buf380, (1, 972, 1, 1), (972, 1, 1, 1), 0), reinterpret_tensor(buf373, (1, 162, 1, 1), (162, 1, 1, 1), 0), reinterpret_tensor(buf358, (1, 906, 1, 1), (906, 1, 1, 1), 0), buf449, reinterpret_tensor(buf351, (1, 906, 1, 1), (906, 1, 1, 1), 0), reinterpret_tensor(buf344, (1, 151, 1, 1), (151, 1, 1, 1), 0), reinterpret_tensor(buf329, (1, 840, 1, 1), (840, 1, 1, 1), 0), buf450, reinterpret_tensor(buf322, (1, 840, 1, 1), (840, 1, 1, 1), 0), reinterpret_tensor(buf316, (1, 140, 1, 1), (140, 1, 1, 1), 0), reinterpret_tensor(buf301, (1, 768, 1, 1), (768, 1, 1, 1), 0), buf451, reinterpret_tensor(buf294, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf287, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf272, (1, 702, 1, 1), (702, 1, 1, 1), 0), buf452, reinterpret_tensor(buf265, (1, 702, 1, 1), (702, 1, 1, 1), 0), reinterpret_tensor(buf258, (1, 117, 1, 1), (117, 1, 1, 1), 0), reinterpret_tensor(buf243, (1, 636, 1, 1), (636, 1, 1, 1), 0), buf453, reinterpret_tensor(buf236, (1, 636, 1, 1), (636, 1, 1, 1), 0), reinterpret_tensor(buf229, (1, 106, 1, 1), (106, 1, 1, 1), 0), reinterpret_tensor(buf214, (1, 570, 1, 1), (570, 1, 1, 1), 0), buf454, reinterpret_tensor(buf207, (1, 570, 1, 1), (570, 1, 1, 1), 0), reinterpret_tensor(buf200, (1, 95, 1, 1), (95, 1, 1, 1), 0), reinterpret_tensor(buf185, (1, 504, 1, 1), (504, 1, 1, 1), 0), buf455, reinterpret_tensor(buf178, (1, 504, 1, 1), (504, 1, 1, 1), 0), reinterpret_tensor(buf171, (1, 84, 1, 1), (84, 1, 1, 1), 0), reinterpret_tensor(buf156, (1, 432, 1, 1), (432, 1, 1, 1), 0), buf456, reinterpret_tensor(buf149, (1, 432, 1, 1), (432, 1, 1, 1), 0), reinterpret_tensor(buf143, (1, 72, 1, 1), (72, 1, 1, 1), 0), reinterpret_tensor(buf128, (1, 366, 1, 1), (366, 1, 1, 1), 0), buf457, reinterpret_tensor(buf121, (1, 366, 1, 1), (366, 1, 1, 1), 0), reinterpret_tensor(buf114, (1, 61, 1, 1), (61, 1, 1, 1), 0), reinterpret_tensor(buf99, (1, 300, 1, 1), (300, 1, 1, 1), 0), buf458, reinterpret_tensor(buf92, (1, 300, 1, 1), (300, 1, 1, 1), 0), reinterpret_tensor(buf86, (1, 50, 1, 1), (50, 1, 1, 1), 0), reinterpret_tensor(buf71, (1, 228, 1, 1), (228, 1, 1, 1), 0), buf459, reinterpret_tensor(buf64, (1, 228, 1, 1), (228, 1, 1, 1), 0), reinterpret_tensor(buf57, (1, 38, 1, 1), (38, 1, 1, 1), 0), buf460, reinterpret_tensor(buf50, (1, 162, 1, 1), (162, 1, 1, 1), 0), buf461, reinterpret_tensor(buf43, (1, 162, 1, 1), (162, 1, 1, 1), 0), reinterpret_tensor(buf37, (1, 27, 1, 1), (27, 1, 1, 1), 0), buf462, reinterpret_tensor(buf30, (1, 96, 1, 1), (96, 1, 1, 1), 0), buf463, reinterpret_tensor(buf23, (1, 96, 1, 1), (96, 1, 1, 1), 0), reinterpret_tensor(buf17, (1, 16, 1, 1), (16, 1, 1, 1), 0), buf464, reinterpret_tensor(buf10, (1, 32, 1, 1), (32, 1, 1, 1), 0), buf465, reinterpret_tensor(buf3, (1, 32, 1, 1), (32, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((27, ), (1, ), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((27, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((162, ), (1, ), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((162, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((162, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((162, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((38, ), (1, ), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((38, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((228, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((228, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((228, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((228, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((50, ), (1, ), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((50, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((300, ), (1, ), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((300, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((300, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((300, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((61, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((61, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((366, ), (1, ), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((366, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((366, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((366, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((432, ), (1, ), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((432, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((432, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((432, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((84, ), (1, ), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((84, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((504, ), (1, ), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((504, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((504, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((504, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((95, ), (1, ), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((95, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((570, ), (1, ), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((570, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((570, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((570, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((106, ), (1, ), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((106, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((636, ), (1, ), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((636, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((636, ), (1, ), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((636, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((117, ), (1, ), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((117, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((702, ), (1, ), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((702, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((702, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((702, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((140, ), (1, ), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((140, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((840, ), (1, ), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((840, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((840, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((840, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((151, ), (1, ), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((151, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((906, ), (1, ), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((906, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((906, ), (1, ), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((906, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((162, ), (1, ), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((162, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((972, ), (1, ), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((972, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((972, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((972, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((174, ), (1, ), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((174, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((1044, ), (1, ), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((1044, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((1044, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((1044, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((185, ), (1, ), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((185, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((27, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((162, 27, 1, 1), (27, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((162, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((38, 162, 1, 1), (162, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((228, 38, 1, 1), (38, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((228, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((19, 228, 1, 1), (228, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((19, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((19, ), (1, ), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((19, ), (1, ), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((228, 19, 1, 1), (19, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((228, ), (1, ), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((50, 228, 1, 1), (228, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((300, 50, 1, 1), (50, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((300, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((25, 300, 1, 1), (300, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((25, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((25, ), (1, ), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((25, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((300, 25, 1, 1), (25, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((300, ), (1, ), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((61, 300, 1, 1), (300, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((366, 61, 1, 1), (61, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((366, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((30, 366, 1, 1), (366, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((30, ), (1, ), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((30, ), (1, ), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((30, ), (1, ), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((366, 30, 1, 1), (30, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((366, ), (1, ), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((72, 366, 1, 1), (366, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((432, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((36, 432, 1, 1), (432, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((432, 36, 1, 1), (36, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((432, ), (1, ), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((84, 432, 1, 1), (432, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((504, 84, 1, 1), (84, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((504, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((42, 504, 1, 1), (504, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((42, ), (1, ), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((42, ), (1, ), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((42, ), (1, ), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((504, 42, 1, 1), (42, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((504, ), (1, ), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((95, 504, 1, 1), (504, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((570, 95, 1, 1), (95, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((570, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((47, 570, 1, 1), (570, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((47, ), (1, ), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((47, ), (1, ), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((47, ), (1, ), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((570, 47, 1, 1), (47, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((570, ), (1, ), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((106, 570, 1, 1), (570, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((636, 106, 1, 1), (106, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((636, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((53, 636, 1, 1), (636, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((53, ), (1, ), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((53, ), (1, ), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((53, ), (1, ), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((636, 53, 1, 1), (53, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((636, ), (1, ), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((117, 636, 1, 1), (636, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((702, 117, 1, 1), (117, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((702, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((58, 702, 1, 1), (702, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((702, 58, 1, 1), (58, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((702, ), (1, ), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((128, 702, 1, 1), (702, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((768, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((64, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((768, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((140, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((840, 140, 1, 1), (140, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((840, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((70, 840, 1, 1), (840, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((70, ), (1, ), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((70, ), (1, ), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((70, ), (1, ), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((840, 70, 1, 1), (70, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((840, ), (1, ), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((151, 840, 1, 1), (840, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((906, 151, 1, 1), (151, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((906, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((75, 906, 1, 1), (906, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((75, ), (1, ), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((75, ), (1, ), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((75, ), (1, ), device='cpu', dtype=torch.float32)
    primals_204 = rand_strided((906, 75, 1, 1), (75, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((906, ), (1, ), device='cpu', dtype=torch.float32)
    primals_206 = rand_strided((162, 906, 1, 1), (906, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_207 = rand_strided((972, 162, 1, 1), (162, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((972, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_209 = rand_strided((81, 972, 1, 1), (972, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_210 = rand_strided((81, ), (1, ), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((81, ), (1, ), device='cpu', dtype=torch.float32)
    primals_212 = rand_strided((81, ), (1, ), device='cpu', dtype=torch.float32)
    primals_213 = rand_strided((972, 81, 1, 1), (81, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_214 = rand_strided((972, ), (1, ), device='cpu', dtype=torch.float32)
    primals_215 = rand_strided((174, 972, 1, 1), (972, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_216 = rand_strided((1044, 174, 1, 1), (174, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((1044, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_218 = rand_strided((87, 1044, 1, 1), (1044, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_219 = rand_strided((87, ), (1, ), device='cpu', dtype=torch.float32)
    primals_220 = rand_strided((87, ), (1, ), device='cpu', dtype=torch.float32)
    primals_221 = rand_strided((87, ), (1, ), device='cpu', dtype=torch.float32)
    primals_222 = rand_strided((1044, 87, 1, 1), (87, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_223 = rand_strided((1044, ), (1, ), device='cpu', dtype=torch.float32)
    primals_224 = rand_strided((185, 1044, 1, 1), (1044, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_225 = rand_strided((1280, 185, 1, 1), (185, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_226 = rand_strided((1000, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    primals_227 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    primals_228 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_229 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_230 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_231 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_232 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_233 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_234 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_235 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_236 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_237 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_238 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_239 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_240 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_241 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_242 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_243 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_244 = rand_strided((27, ), (1, ), device='cpu', dtype=torch.float32)
    primals_245 = rand_strided((27, ), (1, ), device='cpu', dtype=torch.float32)
    primals_246 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_247 = rand_strided((162, ), (1, ), device='cpu', dtype=torch.float32)
    primals_248 = rand_strided((162, ), (1, ), device='cpu', dtype=torch.float32)
    primals_249 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_250 = rand_strided((162, ), (1, ), device='cpu', dtype=torch.float32)
    primals_251 = rand_strided((162, ), (1, ), device='cpu', dtype=torch.float32)
    primals_252 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_253 = rand_strided((38, ), (1, ), device='cpu', dtype=torch.float32)
    primals_254 = rand_strided((38, ), (1, ), device='cpu', dtype=torch.float32)
    primals_255 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_256 = rand_strided((228, ), (1, ), device='cpu', dtype=torch.float32)
    primals_257 = rand_strided((228, ), (1, ), device='cpu', dtype=torch.float32)
    primals_258 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_259 = rand_strided((228, ), (1, ), device='cpu', dtype=torch.float32)
    primals_260 = rand_strided((228, ), (1, ), device='cpu', dtype=torch.float32)
    primals_261 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_262 = rand_strided((50, ), (1, ), device='cpu', dtype=torch.float32)
    primals_263 = rand_strided((50, ), (1, ), device='cpu', dtype=torch.float32)
    primals_264 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_265 = rand_strided((300, ), (1, ), device='cpu', dtype=torch.float32)
    primals_266 = rand_strided((300, ), (1, ), device='cpu', dtype=torch.float32)
    primals_267 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_268 = rand_strided((300, ), (1, ), device='cpu', dtype=torch.float32)
    primals_269 = rand_strided((300, ), (1, ), device='cpu', dtype=torch.float32)
    primals_270 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_271 = rand_strided((61, ), (1, ), device='cpu', dtype=torch.float32)
    primals_272 = rand_strided((61, ), (1, ), device='cpu', dtype=torch.float32)
    primals_273 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_274 = rand_strided((366, ), (1, ), device='cpu', dtype=torch.float32)
    primals_275 = rand_strided((366, ), (1, ), device='cpu', dtype=torch.float32)
    primals_276 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_277 = rand_strided((366, ), (1, ), device='cpu', dtype=torch.float32)
    primals_278 = rand_strided((366, ), (1, ), device='cpu', dtype=torch.float32)
    primals_279 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_280 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_281 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_282 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_283 = rand_strided((432, ), (1, ), device='cpu', dtype=torch.float32)
    primals_284 = rand_strided((432, ), (1, ), device='cpu', dtype=torch.float32)
    primals_285 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_286 = rand_strided((432, ), (1, ), device='cpu', dtype=torch.float32)
    primals_287 = rand_strided((432, ), (1, ), device='cpu', dtype=torch.float32)
    primals_288 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_289 = rand_strided((84, ), (1, ), device='cpu', dtype=torch.float32)
    primals_290 = rand_strided((84, ), (1, ), device='cpu', dtype=torch.float32)
    primals_291 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_292 = rand_strided((504, ), (1, ), device='cpu', dtype=torch.float32)
    primals_293 = rand_strided((504, ), (1, ), device='cpu', dtype=torch.float32)
    primals_294 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_295 = rand_strided((504, ), (1, ), device='cpu', dtype=torch.float32)
    primals_296 = rand_strided((504, ), (1, ), device='cpu', dtype=torch.float32)
    primals_297 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_298 = rand_strided((95, ), (1, ), device='cpu', dtype=torch.float32)
    primals_299 = rand_strided((95, ), (1, ), device='cpu', dtype=torch.float32)
    primals_300 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_301 = rand_strided((570, ), (1, ), device='cpu', dtype=torch.float32)
    primals_302 = rand_strided((570, ), (1, ), device='cpu', dtype=torch.float32)
    primals_303 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_304 = rand_strided((570, ), (1, ), device='cpu', dtype=torch.float32)
    primals_305 = rand_strided((570, ), (1, ), device='cpu', dtype=torch.float32)
    primals_306 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_307 = rand_strided((106, ), (1, ), device='cpu', dtype=torch.float32)
    primals_308 = rand_strided((106, ), (1, ), device='cpu', dtype=torch.float32)
    primals_309 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_310 = rand_strided((636, ), (1, ), device='cpu', dtype=torch.float32)
    primals_311 = rand_strided((636, ), (1, ), device='cpu', dtype=torch.float32)
    primals_312 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_313 = rand_strided((636, ), (1, ), device='cpu', dtype=torch.float32)
    primals_314 = rand_strided((636, ), (1, ), device='cpu', dtype=torch.float32)
    primals_315 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_316 = rand_strided((117, ), (1, ), device='cpu', dtype=torch.float32)
    primals_317 = rand_strided((117, ), (1, ), device='cpu', dtype=torch.float32)
    primals_318 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_319 = rand_strided((702, ), (1, ), device='cpu', dtype=torch.float32)
    primals_320 = rand_strided((702, ), (1, ), device='cpu', dtype=torch.float32)
    primals_321 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_322 = rand_strided((702, ), (1, ), device='cpu', dtype=torch.float32)
    primals_323 = rand_strided((702, ), (1, ), device='cpu', dtype=torch.float32)
    primals_324 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_325 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_326 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_327 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_328 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_329 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_330 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_331 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_332 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_333 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_334 = rand_strided((140, ), (1, ), device='cpu', dtype=torch.float32)
    primals_335 = rand_strided((140, ), (1, ), device='cpu', dtype=torch.float32)
    primals_336 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_337 = rand_strided((840, ), (1, ), device='cpu', dtype=torch.float32)
    primals_338 = rand_strided((840, ), (1, ), device='cpu', dtype=torch.float32)
    primals_339 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_340 = rand_strided((840, ), (1, ), device='cpu', dtype=torch.float32)
    primals_341 = rand_strided((840, ), (1, ), device='cpu', dtype=torch.float32)
    primals_342 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_343 = rand_strided((151, ), (1, ), device='cpu', dtype=torch.float32)
    primals_344 = rand_strided((151, ), (1, ), device='cpu', dtype=torch.float32)
    primals_345 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_346 = rand_strided((906, ), (1, ), device='cpu', dtype=torch.float32)
    primals_347 = rand_strided((906, ), (1, ), device='cpu', dtype=torch.float32)
    primals_348 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_349 = rand_strided((906, ), (1, ), device='cpu', dtype=torch.float32)
    primals_350 = rand_strided((906, ), (1, ), device='cpu', dtype=torch.float32)
    primals_351 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_352 = rand_strided((162, ), (1, ), device='cpu', dtype=torch.float32)
    primals_353 = rand_strided((162, ), (1, ), device='cpu', dtype=torch.float32)
    primals_354 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_355 = rand_strided((972, ), (1, ), device='cpu', dtype=torch.float32)
    primals_356 = rand_strided((972, ), (1, ), device='cpu', dtype=torch.float32)
    primals_357 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_358 = rand_strided((972, ), (1, ), device='cpu', dtype=torch.float32)
    primals_359 = rand_strided((972, ), (1, ), device='cpu', dtype=torch.float32)
    primals_360 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_361 = rand_strided((174, ), (1, ), device='cpu', dtype=torch.float32)
    primals_362 = rand_strided((174, ), (1, ), device='cpu', dtype=torch.float32)
    primals_363 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_364 = rand_strided((1044, ), (1, ), device='cpu', dtype=torch.float32)
    primals_365 = rand_strided((1044, ), (1, ), device='cpu', dtype=torch.float32)
    primals_366 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_367 = rand_strided((1044, ), (1, ), device='cpu', dtype=torch.float32)
    primals_368 = rand_strided((1044, ), (1, ), device='cpu', dtype=torch.float32)
    primals_369 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_370 = rand_strided((185, ), (1, ), device='cpu', dtype=torch.float32)
    primals_371 = rand_strided((185, ), (1, ), device='cpu', dtype=torch.float32)
    primals_372 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_373 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_374 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_375 = rand_strided((19, ), (1, ), device='cpu', dtype=torch.float32)
    primals_376 = rand_strided((19, ), (1, ), device='cpu', dtype=torch.float32)
    primals_377 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_378 = rand_strided((25, ), (1, ), device='cpu', dtype=torch.float32)
    primals_379 = rand_strided((25, ), (1, ), device='cpu', dtype=torch.float32)
    primals_380 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_381 = rand_strided((30, ), (1, ), device='cpu', dtype=torch.float32)
    primals_382 = rand_strided((30, ), (1, ), device='cpu', dtype=torch.float32)
    primals_383 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_384 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    primals_385 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    primals_386 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_387 = rand_strided((42, ), (1, ), device='cpu', dtype=torch.float32)
    primals_388 = rand_strided((42, ), (1, ), device='cpu', dtype=torch.float32)
    primals_389 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_390 = rand_strided((47, ), (1, ), device='cpu', dtype=torch.float32)
    primals_391 = rand_strided((47, ), (1, ), device='cpu', dtype=torch.float32)
    primals_392 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_393 = rand_strided((53, ), (1, ), device='cpu', dtype=torch.float32)
    primals_394 = rand_strided((53, ), (1, ), device='cpu', dtype=torch.float32)
    primals_395 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_396 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_397 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_398 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_399 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_400 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_401 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_402 = rand_strided((70, ), (1, ), device='cpu', dtype=torch.float32)
    primals_403 = rand_strided((70, ), (1, ), device='cpu', dtype=torch.float32)
    primals_404 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_405 = rand_strided((75, ), (1, ), device='cpu', dtype=torch.float32)
    primals_406 = rand_strided((75, ), (1, ), device='cpu', dtype=torch.float32)
    primals_407 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_408 = rand_strided((81, ), (1, ), device='cpu', dtype=torch.float32)
    primals_409 = rand_strided((81, ), (1, ), device='cpu', dtype=torch.float32)
    primals_410 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_411 = rand_strided((87, ), (1, ), device='cpu', dtype=torch.float32)
    primals_412 = rand_strided((87, ), (1, ), device='cpu', dtype=torch.float32)
    primals_413 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_414 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('rexnet_100', benchmark_compiled_module)
