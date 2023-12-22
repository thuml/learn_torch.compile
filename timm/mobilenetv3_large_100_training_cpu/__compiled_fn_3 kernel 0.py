
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_functional_hardswish_1 = async_compile.cpp('''
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1605632L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_2 = async_compile.cpp('''
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
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_3 = async_compile.cpp('''
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
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (16L*x0)));
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
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_4 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
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
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
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
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
                }
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (24L*x1)));
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
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (24L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (72L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (72L*x0)));
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
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (72L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_8 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (72L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (72L*x0)));
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
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (72L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_9 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (24L*x1)));
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
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (24L*x0)));
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
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_10 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (72L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (72L*x0)));
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
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (72L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_mean_relu_11 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (72L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
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
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (72L*x0)));
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
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (72L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1 + (72L*x2) + (56448L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x1 + (72L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(576L); x0+=static_cast<long>(8L))
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


cpp_fused_relu_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_hardsigmoid_mul_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(576L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(3.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 + tmp2;
            auto tmp4 = static_cast<float>(0.0);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = at::vec::maximum(tmp3, tmp5);
            auto tmp7 = static_cast<float>(6.0);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = at::vec::minimum(tmp6, tmp8);
            auto tmp10 = tmp9 / tmp8;
            tmp10.store(out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(72L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (72L*x1) + (56448L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (72L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp2.store(out_ptr1 + static_cast<long>(x2 + (72L*x1) + (56448L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (40L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
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
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (40L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (40L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (120L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
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
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (120L*x0)));
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
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_mean_relu_16 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (120L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
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
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (120L*x0)));
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
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1 + (120L*x2) + (94080L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x1 + (120L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
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


cpp_fused_relu_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_hardsigmoid_mul_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(3.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 + tmp2;
            auto tmp4 = static_cast<float>(0.0);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = at::vec::maximum(tmp3, tmp5);
            auto tmp7 = static_cast<float>(6.0);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = at::vec::minimum(tmp6, tmp8);
            auto tmp10 = tmp9 / tmp8;
            tmp10.store(out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(120L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (120L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp2.store(out_ptr1 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_19 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (40L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
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
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (40L*x0)));
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
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (40L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_20 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (120L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
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
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (120L*x0)));
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
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_mean_relu_21 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (120L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
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
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (120L*x0)));
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
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1 + (120L*x2) + (94080L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x1 + (120L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
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


cpp_fused_relu_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_hardsigmoid_mul_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(3.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 + tmp2;
            auto tmp4 = static_cast<float>(0.0);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = at::vec::maximum(tmp3, tmp5);
            auto tmp7 = static_cast<float>(6.0);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = at::vec::minimum(tmp6, tmp8);
            auto tmp10 = tmp9 / tmp8;
            tmp10.store(out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(120L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (120L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp2.store(out_ptr1 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_24 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (40L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
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
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (40L*x0)));
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
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (40L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_hardswish_25 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (240L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
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
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (240L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1505280L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_hardswish_26 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (240L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (240L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(376320L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_27 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (80L*x1)));
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
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (80L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (80L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_hardswish_28 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (200L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(200L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(200L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (200L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (200L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(313600L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_hardswish_29 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (200L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(200L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(200L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (200L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (200L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(313600L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_30 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (80L*x1)));
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
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (80L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (80L*x0)));
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
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (80L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_hardswish_31 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (184L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(184L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (184L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (184L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(288512L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_hardswish_32 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (184L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(184L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (184L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (184L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(288512L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_33 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (80L*x1)));
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
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (80L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (80L*x0)));
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
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (80L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_hardswish_34 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (184L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(184L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (184L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (184L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(288512L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_hardswish_35 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (184L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(184L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (184L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (184L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(288512L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_36 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (80L*x1)));
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
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (80L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (80L*x0)));
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
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (80L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_hardswish_37 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (480L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (480L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(752640L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_hardswish_mean_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    auto out_ptr5 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (480L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (480L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(752640L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(out_ptr4 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr5 + static_cast<long>(x1 + (480L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3840L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
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


cpp_fused_relu_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_hardsigmoid_mul_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3840L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(3.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 + tmp2;
            auto tmp4 = static_cast<float>(0.0);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = at::vec::maximum(tmp3, tmp5);
            auto tmp7 = static_cast<float>(6.0);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = at::vec::minimum(tmp6, tmp8);
            auto tmp10 = tmp9 / tmp8;
            tmp10.store(out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(480L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (480L*x1) + (94080L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (480L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp2.store(out_ptr1 + static_cast<long>(x2 + (480L*x1) + (94080L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_41 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (112L*x1)));
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
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (112L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (112L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_hardswish_42 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (672L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (672L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1053696L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_hardswish_mean_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    auto out_ptr5 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (672L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (672L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1053696L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(out_ptr4 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1 + (672L*x2) + (131712L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr5 + static_cast<long>(x1 + (672L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5376L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
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


cpp_fused_relu_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1344L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_hardsigmoid_mul_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(5376L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(3.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 + tmp2;
            auto tmp4 = static_cast<float>(0.0);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = at::vec::maximum(tmp3, tmp5);
            auto tmp7 = static_cast<float>(6.0);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = at::vec::minimum(tmp6, tmp8);
            auto tmp10 = tmp9 / tmp8;
            tmp10.store(out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(672L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (672L*x1) + (131712L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp2.store(out_ptr1 + static_cast<long>(x2 + (672L*x1) + (131712L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_46 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (112L*x1)));
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
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (112L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (112L*x0)));
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
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (112L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_hardswish_47 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (672L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (672L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1053696L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_hardswish_mean_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    auto out_ptr5 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (672L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (672L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(263424L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(out_ptr4 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1 + (672L*x2) + (32928L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr5 + static_cast<long>(x1 + (672L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5376L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
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


cpp_fused_relu_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1344L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_hardsigmoid_mul_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(5376L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(3.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 + tmp2;
            auto tmp4 = static_cast<float>(0.0);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = at::vec::maximum(tmp3, tmp5);
            auto tmp7 = static_cast<float>(6.0);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = at::vec::minimum(tmp6, tmp8);
            auto tmp10 = tmp9 / tmp8;
            tmp10.store(out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(672L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (672L*x1) + (32928L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp2.store(out_ptr1 + static_cast<long>(x2 + (672L*x1) + (32928L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_51 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
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
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
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
                tmp15.store(out_ptr3 + static_cast<long>(x1 + (160L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_hardswish_52 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (960L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (960L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (960L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(376320L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_hardswish_mean_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    auto out_ptr5 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (960L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (960L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (960L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(376320L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(out_ptr4 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1 + (960L*x2) + (47040L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr5 + static_cast<long>(x1 + (960L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(7680L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
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


cpp_fused_relu_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_hardsigmoid_hardsigmoid_backward_mul_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       bool* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(7680L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr0[static_cast<long>(x0)];
            auto tmp1 = static_cast<float>(3.0);
            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
            auto tmp3 = static_cast<float>(0.0);
            auto tmp4 = max_propagate_nan(tmp2, tmp3);
            auto tmp5 = static_cast<float>(6.0);
            auto tmp6 = min_propagate_nan(tmp4, tmp5);
            auto tmp7 = tmp6 / tmp5;
            auto tmp8 = static_cast<float>(-3.0);
            auto tmp9 = tmp0 > tmp8;
            auto tmp10 = tmp0 < tmp1;
            auto tmp11 = decltype(tmp9)(tmp9 & tmp10);
            out_ptr0[static_cast<long>(x0)] = tmp7;
            out_ptr1[static_cast<long>(x0)] = tmp11;
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(960L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (960L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp2.store(out_ptr2 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_56 = async_compile.cpp('''
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
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
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (160L*x0)));
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
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr3 + static_cast<long>(x1 + (160L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_hardswish_57 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (960L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (960L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (960L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(376320L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_hardswish_mean_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    auto out_ptr5 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (960L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (960L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (960L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(376320L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(out_ptr4 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1 + (960L*x2) + (47040L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr5 + static_cast<long>(x1 + (960L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(7680L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
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


cpp_fused_relu_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_hardsigmoid_hardsigmoid_backward_mul_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       bool* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(7680L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr0[static_cast<long>(x0)];
            auto tmp1 = static_cast<float>(3.0);
            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
            auto tmp3 = static_cast<float>(0.0);
            auto tmp4 = max_propagate_nan(tmp2, tmp3);
            auto tmp5 = static_cast<float>(6.0);
            auto tmp6 = min_propagate_nan(tmp4, tmp5);
            auto tmp7 = tmp6 / tmp5;
            auto tmp8 = static_cast<float>(-3.0);
            auto tmp9 = tmp0 > tmp8;
            auto tmp10 = tmp0 < tmp1;
            auto tmp11 = decltype(tmp9)(tmp9 & tmp10);
            out_ptr0[static_cast<long>(x0)] = tmp7;
            out_ptr1[static_cast<long>(x0)] = tmp11;
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(960L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (960L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp2.store(out_ptr2 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_61 = async_compile.cpp('''
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
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
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (160L*x0)));
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
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr3 + static_cast<long>(x1 + (160L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_hardswish_mean_62 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (960L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (960L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (960L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1 + (960L*x2) + (47040L*x0)));
                            auto tmp1 = static_cast<float>(3.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 + tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = at::vec::maximum(tmp3, tmp5);
                            auto tmp7 = static_cast<float>(6.0);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = at::vec::minimum(tmp6, tmp8);
                            auto tmp10 = tmp0 * tmp9;
                            auto tmp11 = tmp10 / tmp8;
                            tmp_acc0_vec = tmp_acc0_vec + tmp11;
                        }
                        tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x1 + (960L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(7680L); x0+=static_cast<long>(8L))
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


cpp_fused_hardswish_view_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(10240L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(3.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 + tmp2;
            auto tmp4 = static_cast<float>(0.0);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = at::vec::maximum(tmp3, tmp5);
            auto tmp7 = static_cast<float>(6.0);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = at::vec::minimum(tmp6, tmp8);
            auto tmp10 = tmp0 * tmp9;
            auto tmp11 = tmp10 / tmp8;
            tmp11.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_hardsigmoid_backward_64 = async_compile.cpp('''
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
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const long* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       const long* in_ptr10,
                       const float* in_ptr11,
                       const float* in_ptr12,
                       const float* in_ptr13,
                       const long* in_ptr14,
                       const float* in_ptr15,
                       const float* in_ptr16,
                       const float* in_ptr17,
                       const long* in_ptr18,
                       const float* in_ptr19,
                       const float* in_ptr20,
                       const float* in_ptr21,
                       const long* in_ptr22,
                       const float* in_ptr23,
                       const float* in_ptr24,
                       const float* in_ptr25,
                       const long* in_ptr26,
                       const float* in_ptr27,
                       const float* in_ptr28,
                       const float* in_ptr29,
                       const long* in_ptr30,
                       const float* in_ptr31,
                       const float* in_ptr32,
                       const float* in_ptr33,
                       const long* in_ptr34,
                       const float* in_ptr35,
                       const float* in_ptr36,
                       const float* in_ptr37,
                       const long* in_ptr38,
                       const float* in_ptr39,
                       const float* in_ptr40,
                       const float* in_ptr41,
                       const long* in_ptr42,
                       const float* in_ptr43,
                       const float* in_ptr44,
                       const float* in_ptr45,
                       const long* in_ptr46,
                       const float* in_ptr47,
                       const float* in_ptr48,
                       const float* in_ptr49,
                       const long* in_ptr50,
                       const float* in_ptr51,
                       const float* in_ptr52,
                       const float* in_ptr53,
                       const long* in_ptr54,
                       const float* in_ptr55,
                       const float* in_ptr56,
                       const float* in_ptr57,
                       const long* in_ptr58,
                       const float* in_ptr59,
                       const float* in_ptr60,
                       const float* in_ptr61,
                       const long* in_ptr62,
                       const float* in_ptr63,
                       const float* in_ptr64,
                       const float* in_ptr65,
                       const long* in_ptr66,
                       const float* in_ptr67,
                       const float* in_ptr68,
                       const float* in_ptr69,
                       const long* in_ptr70,
                       const float* in_ptr71,
                       const float* in_ptr72,
                       const float* in_ptr73,
                       const long* in_ptr74,
                       const float* in_ptr75,
                       const float* in_ptr76,
                       const float* in_ptr77,
                       const long* in_ptr78,
                       const float* in_ptr79,
                       const float* in_ptr80,
                       const float* in_ptr81,
                       const long* in_ptr82,
                       const float* in_ptr83,
                       const float* in_ptr84,
                       const float* in_ptr85,
                       const long* in_ptr86,
                       const float* in_ptr87,
                       const float* in_ptr88,
                       const float* in_ptr89,
                       const long* in_ptr90,
                       const float* in_ptr91,
                       const float* in_ptr92,
                       const float* in_ptr93,
                       const long* in_ptr94,
                       const float* in_ptr95,
                       const float* in_ptr96,
                       const float* in_ptr97,
                       const long* in_ptr98,
                       const float* in_ptr99,
                       const float* in_ptr100,
                       const float* in_ptr101,
                       const long* in_ptr102,
                       const float* in_ptr103,
                       const float* in_ptr104,
                       const float* in_ptr105,
                       const long* in_ptr106,
                       const float* in_ptr107,
                       const float* in_ptr108,
                       const float* in_ptr109,
                       const long* in_ptr110,
                       const float* in_ptr111,
                       const float* in_ptr112,
                       const float* in_ptr113,
                       const long* in_ptr114,
                       const float* in_ptr115,
                       const float* in_ptr116,
                       const float* in_ptr117,
                       const long* in_ptr118,
                       const float* in_ptr119,
                       const float* in_ptr120,
                       const float* in_ptr121,
                       const long* in_ptr122,
                       const float* in_ptr123,
                       const float* in_ptr124,
                       const float* in_ptr125,
                       const long* in_ptr126,
                       const float* in_ptr127,
                       const float* in_ptr128,
                       const float* in_ptr129,
                       const long* in_ptr130,
                       const float* in_ptr131,
                       const float* in_ptr132,
                       const float* in_ptr133,
                       const long* in_ptr134,
                       const float* in_ptr135,
                       const float* in_ptr136,
                       const float* in_ptr137,
                       const long* in_ptr138,
                       const float* in_ptr139,
                       const float* in_ptr140,
                       const float* in_ptr141,
                       const long* in_ptr142,
                       const float* in_ptr143,
                       const float* in_ptr144,
                       const float* in_ptr145,
                       const long* in_ptr146,
                       const float* in_ptr147,
                       const float* in_ptr148,
                       const float* in_ptr149,
                       const long* in_ptr150,
                       const float* in_ptr151,
                       const float* in_ptr152,
                       const float* in_ptr153,
                       const long* in_ptr154,
                       const float* in_ptr155,
                       const float* in_ptr156,
                       const float* in_ptr157,
                       const long* in_ptr158,
                       const float* in_ptr159,
                       const float* in_ptr160,
                       const float* in_ptr161,
                       const long* in_ptr162,
                       const float* in_ptr163,
                       const float* in_ptr164,
                       const float* in_ptr165,
                       const long* in_ptr166,
                       const float* in_ptr167,
                       const float* in_ptr168,
                       const float* in_ptr169,
                       const long* in_ptr170,
                       const float* in_ptr171,
                       const float* in_ptr172,
                       const float* in_ptr173,
                       const long* in_ptr174,
                       const float* in_ptr175,
                       const float* in_ptr176,
                       const float* in_ptr177,
                       const long* in_ptr178,
                       const float* in_ptr179,
                       const float* in_ptr180,
                       const float* in_ptr181,
                       const long* in_ptr182,
                       const float* in_ptr183,
                       const float* in_ptr184,
                       const float* in_ptr185,
                       const long* in_ptr186,
                       const float* in_ptr187,
                       const float* in_ptr188,
                       const float* in_ptr189,
                       bool* out_ptr0,
                       bool* out_ptr1,
                       bool* out_ptr2,
                       bool* out_ptr3,
                       bool* out_ptr4,
                       bool* out_ptr5,
                       long* out_ptr7,
                       float* out_ptr9,
                       float* out_ptr10,
                       long* out_ptr12,
                       float* out_ptr14,
                       float* out_ptr15,
                       long* out_ptr17,
                       float* out_ptr19,
                       float* out_ptr20,
                       long* out_ptr22,
                       float* out_ptr24,
                       float* out_ptr25,
                       long* out_ptr27,
                       float* out_ptr29,
                       float* out_ptr30,
                       long* out_ptr32,
                       float* out_ptr34,
                       float* out_ptr35,
                       long* out_ptr37,
                       float* out_ptr39,
                       float* out_ptr40,
                       long* out_ptr42,
                       float* out_ptr44,
                       float* out_ptr45,
                       long* out_ptr47,
                       float* out_ptr49,
                       float* out_ptr50,
                       long* out_ptr52,
                       float* out_ptr54,
                       float* out_ptr55,
                       long* out_ptr57,
                       float* out_ptr59,
                       float* out_ptr60,
                       long* out_ptr62,
                       float* out_ptr64,
                       float* out_ptr65,
                       long* out_ptr67,
                       float* out_ptr69,
                       float* out_ptr70,
                       long* out_ptr72,
                       float* out_ptr74,
                       float* out_ptr75,
                       long* out_ptr77,
                       float* out_ptr79,
                       float* out_ptr80,
                       long* out_ptr82,
                       float* out_ptr84,
                       float* out_ptr85,
                       long* out_ptr87,
                       float* out_ptr89,
                       float* out_ptr90,
                       long* out_ptr92,
                       float* out_ptr94,
                       float* out_ptr95,
                       long* out_ptr97,
                       float* out_ptr99,
                       float* out_ptr100,
                       long* out_ptr102,
                       float* out_ptr104,
                       float* out_ptr105,
                       long* out_ptr107,
                       float* out_ptr109,
                       float* out_ptr110,
                       long* out_ptr112,
                       float* out_ptr114,
                       float* out_ptr115,
                       long* out_ptr117,
                       float* out_ptr119,
                       float* out_ptr120,
                       long* out_ptr122,
                       float* out_ptr124,
                       float* out_ptr125,
                       long* out_ptr127,
                       float* out_ptr129,
                       float* out_ptr130,
                       long* out_ptr132,
                       float* out_ptr134,
                       float* out_ptr135,
                       long* out_ptr137,
                       float* out_ptr139,
                       float* out_ptr140,
                       long* out_ptr142,
                       float* out_ptr144,
                       float* out_ptr145,
                       long* out_ptr147,
                       float* out_ptr149,
                       float* out_ptr150,
                       long* out_ptr152,
                       float* out_ptr154,
                       float* out_ptr155,
                       long* out_ptr157,
                       float* out_ptr159,
                       float* out_ptr160,
                       long* out_ptr162,
                       float* out_ptr164,
                       float* out_ptr165,
                       long* out_ptr167,
                       float* out_ptr169,
                       float* out_ptr170,
                       long* out_ptr172,
                       float* out_ptr174,
                       float* out_ptr175,
                       long* out_ptr177,
                       float* out_ptr179,
                       float* out_ptr180,
                       long* out_ptr182,
                       float* out_ptr184,
                       float* out_ptr185,
                       long* out_ptr187,
                       float* out_ptr189,
                       float* out_ptr190,
                       long* out_ptr192,
                       float* out_ptr194,
                       float* out_ptr195,
                       long* out_ptr197,
                       float* out_ptr199,
                       float* out_ptr200,
                       long* out_ptr202,
                       float* out_ptr204,
                       float* out_ptr205,
                       long* out_ptr207,
                       float* out_ptr209,
                       float* out_ptr210,
                       long* out_ptr212,
                       float* out_ptr214,
                       float* out_ptr215,
                       long* out_ptr217,
                       float* out_ptr219,
                       float* out_ptr220,
                       long* out_ptr222,
                       float* out_ptr224,
                       float* out_ptr225,
                       long* out_ptr227,
                       float* out_ptr229,
                       float* out_ptr230,
                       long* out_ptr232,
                       float* out_ptr234,
                       float* out_ptr235)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(5376L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr0[static_cast<long>(x0)];
            auto tmp1 = static_cast<float>(-3.0);
            auto tmp2 = tmp0 > tmp1;
            auto tmp3 = static_cast<float>(3.0);
            auto tmp4 = tmp0 < tmp3;
            auto tmp5 = decltype(tmp2)(tmp2 & tmp4);
            out_ptr0[static_cast<long>(x0)] = tmp5;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(5376L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr1[static_cast<long>(x0)];
            auto tmp1 = static_cast<float>(-3.0);
            auto tmp2 = tmp0 > tmp1;
            auto tmp3 = static_cast<float>(3.0);
            auto tmp4 = tmp0 < tmp3;
            auto tmp5 = decltype(tmp2)(tmp2 & tmp4);
            out_ptr1[static_cast<long>(x0)] = tmp5;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3840L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr2[static_cast<long>(x0)];
            auto tmp1 = static_cast<float>(-3.0);
            auto tmp2 = tmp0 > tmp1;
            auto tmp3 = static_cast<float>(3.0);
            auto tmp4 = tmp0 < tmp3;
            auto tmp5 = decltype(tmp2)(tmp2 & tmp4);
            out_ptr2[static_cast<long>(x0)] = tmp5;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr3[static_cast<long>(x0)];
            auto tmp1 = static_cast<float>(-3.0);
            auto tmp2 = tmp0 > tmp1;
            auto tmp3 = static_cast<float>(3.0);
            auto tmp4 = tmp0 < tmp3;
            auto tmp5 = decltype(tmp2)(tmp2 & tmp4);
            out_ptr3[static_cast<long>(x0)] = tmp5;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr4[static_cast<long>(x0)];
            auto tmp1 = static_cast<float>(-3.0);
            auto tmp2 = tmp0 > tmp1;
            auto tmp3 = static_cast<float>(3.0);
            auto tmp4 = tmp0 < tmp3;
            auto tmp5 = decltype(tmp2)(tmp2 & tmp4);
            out_ptr4[static_cast<long>(x0)] = tmp5;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(576L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr5[static_cast<long>(x0)];
            auto tmp1 = static_cast<float>(-3.0);
            auto tmp2 = tmp0 > tmp1;
            auto tmp3 = static_cast<float>(3.0);
            auto tmp4 = tmp0 < tmp3;
            auto tmp5 = decltype(tmp2)(tmp2 & tmp4);
            out_ptr5[static_cast<long>(x0)] = tmp5;
        }
    }
    {
        auto tmp0 = in_ptr6[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr7[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr9 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr9 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr10 + static_cast<long>(x0));
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
            tmp14.store(out_ptr10 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr10[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr12[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr14 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr14 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr15 + static_cast<long>(x0));
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
            tmp14.store(out_ptr15 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr14[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr17[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr19 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr19 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr20 + static_cast<long>(x0));
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
            tmp14.store(out_ptr20 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr18[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr22[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr19 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr24 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr24 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr3 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr25 + static_cast<long>(x0));
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
            tmp14.store(out_ptr25 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr22[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr27[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr23 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr29 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr29 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr4 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr30 + static_cast<long>(x0));
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
            tmp14.store(out_ptr30 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr26[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr32[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr27 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr34 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr34 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr5 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr35 + static_cast<long>(x0));
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
            tmp14.store(out_ptr35 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr30[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr37[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr31 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr39 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr39 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr6 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr40 + static_cast<long>(x0));
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
            tmp14.store(out_ptr40 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr34[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr42[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr35 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr44 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr44 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr7 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr45 + static_cast<long>(x0));
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
            tmp14.store(out_ptr45 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr38[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr47[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr39 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr49 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr49 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr8 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr50 + static_cast<long>(x0));
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
            tmp14.store(out_ptr50 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr42[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr52[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr43 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr54 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr54 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr9 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr55 + static_cast<long>(x0));
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
            tmp14.store(out_ptr55 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr46[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr57[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr47 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr59 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr59 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr10 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr60 + static_cast<long>(x0));
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
            tmp14.store(out_ptr60 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr50[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr62[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr51 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr64 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr64 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr11 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr65 + static_cast<long>(x0));
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
            tmp14.store(out_ptr65 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr54[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr67[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr55 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr69 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr69 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr12 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr70 + static_cast<long>(x0));
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
            tmp14.store(out_ptr70 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr58[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr72[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr59 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr74 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr74 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr13 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr75 + static_cast<long>(x0));
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
            tmp14.store(out_ptr75 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr62[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr77[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr63 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr79 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr79 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr14 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr80 + static_cast<long>(x0));
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
            tmp14.store(out_ptr80 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr66[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr82[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr67 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr84 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr84 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr15 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr85 + static_cast<long>(x0));
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
            tmp14.store(out_ptr85 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr70[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr87[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr71 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr89 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr89 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr16 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr90 + static_cast<long>(x0));
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
            tmp14.store(out_ptr90 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr74[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr92[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr75 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr94 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr94 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr17 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr95 + static_cast<long>(x0));
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
            tmp14.store(out_ptr95 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr78[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr97[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr79 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr99 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr99 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr18 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr100 + static_cast<long>(x0));
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
            tmp14.store(out_ptr100 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr82[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr102[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr83 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr104 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr104 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr19 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr105 + static_cast<long>(x0));
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
            tmp14.store(out_ptr105 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr86[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr107[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr87 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr109 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr109 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr20 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr110 + static_cast<long>(x0));
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
            tmp14.store(out_ptr110 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr90[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr112[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(200L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr91 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr114 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr114 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(200L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr21 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr115 + static_cast<long>(x0));
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
            tmp14.store(out_ptr115 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr94[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr117[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(200L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr95 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr119 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr119 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(200L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr22 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr120 + static_cast<long>(x0));
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
            tmp14.store(out_ptr120 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr98[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr122[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr99 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr124 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr124 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr23 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr125 + static_cast<long>(x0));
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
            tmp14.store(out_ptr125 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr102[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr127[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr103 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr129 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr129 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr24 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr130 + static_cast<long>(x0));
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
            tmp14.store(out_ptr130 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr106[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr132[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr107 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr134 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr134 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr25 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr135 + static_cast<long>(x0));
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
            tmp14.store(out_ptr135 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr110[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr137[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr111 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr139 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr139 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr26 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr140 + static_cast<long>(x0));
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
            tmp14.store(out_ptr140 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr114[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr142[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr115 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr144 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr144 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr27 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr145 + static_cast<long>(x0));
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
            tmp14.store(out_ptr145 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr118[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr147[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr119 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr149 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr149 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr28 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr150 + static_cast<long>(x0));
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
            tmp14.store(out_ptr150 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr122[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr152[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr123 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr154 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr154 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr29 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr155 + static_cast<long>(x0));
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
            tmp14.store(out_ptr155 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr126[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr157[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr127 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr159 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr159 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr30 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr160 + static_cast<long>(x0));
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
            tmp14.store(out_ptr160 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr130[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr162[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr131 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr164 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr164 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr31 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr165 + static_cast<long>(x0));
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
            tmp14.store(out_ptr165 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr134[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr167[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr135 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr169 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr169 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr32 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr170 + static_cast<long>(x0));
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
            tmp14.store(out_ptr170 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr138[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr172[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr139 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr174 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr174 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr33 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr175 + static_cast<long>(x0));
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
            tmp14.store(out_ptr175 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr142[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr177[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr143 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr179 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr179 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr34 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr180 + static_cast<long>(x0));
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
            tmp14.store(out_ptr180 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr146[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr182[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr147 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr184 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr184 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr35 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr185 + static_cast<long>(x0));
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
            tmp14.store(out_ptr185 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr150[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr187[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr151 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr189 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr189 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr36 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr190 + static_cast<long>(x0));
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
            tmp14.store(out_ptr190 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr154[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr192[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr155 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr194 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr194 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr37 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr195 + static_cast<long>(x0));
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
            tmp14.store(out_ptr195 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr158[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr197[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr159 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr199 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr199 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr38 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr200 + static_cast<long>(x0));
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
            tmp14.store(out_ptr200 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr162[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr202[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr163 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr204 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr204 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr39 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr205 + static_cast<long>(x0));
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
            tmp14.store(out_ptr205 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr166[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr207[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr167 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr209 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr209 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr40 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr210 + static_cast<long>(x0));
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
            tmp14.store(out_ptr210 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr170[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr212[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr171 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr214 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr214 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr41 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr215 + static_cast<long>(x0));
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
            tmp14.store(out_ptr215 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr174[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr217[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr175 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr219 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr219 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr42 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr220 + static_cast<long>(x0));
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
            tmp14.store(out_ptr220 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr178[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr222[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr179 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr224 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr224 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr43 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr225 + static_cast<long>(x0));
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
            tmp14.store(out_ptr225 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr182[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr227[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr183 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr229 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr229 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr44 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr230 + static_cast<long>(x0));
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
            tmp14.store(out_ptr230 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr186[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr232[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr187 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr234 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr234 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr45 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr235 + static_cast<long>(x0));
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
            tmp14.store(out_ptr235 + static_cast<long>(x0));
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313 = args
    args.clear()
    assert_size_stride(primals_1, (16, ), (1, ))
    assert_size_stride(primals_2, (16, ), (1, ))
    assert_size_stride(primals_3, (16, ), (1, ))
    assert_size_stride(primals_4, (16, ), (1, ))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_6, (16, ), (1, ))
    assert_size_stride(primals_7, (64, ), (1, ))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_10, (64, ), (1, ))
    assert_size_stride(primals_11, (24, ), (1, ))
    assert_size_stride(primals_12, (24, ), (1, ))
    assert_size_stride(primals_13, (72, ), (1, ))
    assert_size_stride(primals_14, (72, ), (1, ))
    assert_size_stride(primals_15, (72, ), (1, ))
    assert_size_stride(primals_16, (72, ), (1, ))
    assert_size_stride(primals_17, (24, ), (1, ))
    assert_size_stride(primals_18, (24, ), (1, ))
    assert_size_stride(primals_19, (72, ), (1, ))
    assert_size_stride(primals_20, (72, ), (1, ))
    assert_size_stride(primals_21, (72, ), (1, ))
    assert_size_stride(primals_22, (72, ), (1, ))
    assert_size_stride(primals_23, (40, ), (1, ))
    assert_size_stride(primals_24, (40, ), (1, ))
    assert_size_stride(primals_25, (120, ), (1, ))
    assert_size_stride(primals_26, (120, ), (1, ))
    assert_size_stride(primals_27, (120, ), (1, ))
    assert_size_stride(primals_28, (120, ), (1, ))
    assert_size_stride(primals_29, (40, ), (1, ))
    assert_size_stride(primals_30, (40, ), (1, ))
    assert_size_stride(primals_31, (120, ), (1, ))
    assert_size_stride(primals_32, (120, ), (1, ))
    assert_size_stride(primals_33, (120, ), (1, ))
    assert_size_stride(primals_34, (120, ), (1, ))
    assert_size_stride(primals_35, (40, ), (1, ))
    assert_size_stride(primals_36, (40, ), (1, ))
    assert_size_stride(primals_37, (240, ), (1, ))
    assert_size_stride(primals_38, (240, ), (1, ))
    assert_size_stride(primals_39, (240, ), (1, ))
    assert_size_stride(primals_40, (240, ), (1, ))
    assert_size_stride(primals_41, (80, ), (1, ))
    assert_size_stride(primals_42, (80, ), (1, ))
    assert_size_stride(primals_43, (200, ), (1, ))
    assert_size_stride(primals_44, (200, ), (1, ))
    assert_size_stride(primals_45, (200, ), (1, ))
    assert_size_stride(primals_46, (200, ), (1, ))
    assert_size_stride(primals_47, (80, ), (1, ))
    assert_size_stride(primals_48, (80, ), (1, ))
    assert_size_stride(primals_49, (184, ), (1, ))
    assert_size_stride(primals_50, (184, ), (1, ))
    assert_size_stride(primals_51, (184, ), (1, ))
    assert_size_stride(primals_52, (184, ), (1, ))
    assert_size_stride(primals_53, (80, ), (1, ))
    assert_size_stride(primals_54, (80, ), (1, ))
    assert_size_stride(primals_55, (184, ), (1, ))
    assert_size_stride(primals_56, (184, ), (1, ))
    assert_size_stride(primals_57, (184, ), (1, ))
    assert_size_stride(primals_58, (184, ), (1, ))
    assert_size_stride(primals_59, (80, ), (1, ))
    assert_size_stride(primals_60, (80, ), (1, ))
    assert_size_stride(primals_61, (480, ), (1, ))
    assert_size_stride(primals_62, (480, ), (1, ))
    assert_size_stride(primals_63, (480, ), (1, ))
    assert_size_stride(primals_64, (480, ), (1, ))
    assert_size_stride(primals_65, (112, ), (1, ))
    assert_size_stride(primals_66, (112, ), (1, ))
    assert_size_stride(primals_67, (672, ), (1, ))
    assert_size_stride(primals_68, (672, ), (1, ))
    assert_size_stride(primals_69, (672, ), (1, ))
    assert_size_stride(primals_70, (672, ), (1, ))
    assert_size_stride(primals_71, (112, ), (1, ))
    assert_size_stride(primals_72, (112, ), (1, ))
    assert_size_stride(primals_73, (672, ), (1, ))
    assert_size_stride(primals_74, (672, ), (1, ))
    assert_size_stride(primals_75, (672, ), (1, ))
    assert_size_stride(primals_76, (672, ), (1, ))
    assert_size_stride(primals_77, (160, ), (1, ))
    assert_size_stride(primals_78, (160, ), (1, ))
    assert_size_stride(primals_79, (960, ), (1, ))
    assert_size_stride(primals_80, (960, ), (1, ))
    assert_size_stride(primals_81, (960, ), (1, ))
    assert_size_stride(primals_82, (960, ), (1, ))
    assert_size_stride(primals_83, (160, ), (1, ))
    assert_size_stride(primals_84, (160, ), (1, ))
    assert_size_stride(primals_85, (960, ), (1, ))
    assert_size_stride(primals_86, (960, ), (1, ))
    assert_size_stride(primals_87, (960, ), (1, ))
    assert_size_stride(primals_88, (960, ), (1, ))
    assert_size_stride(primals_89, (160, ), (1, ))
    assert_size_stride(primals_90, (160, ), (1, ))
    assert_size_stride(primals_91, (960, ), (1, ))
    assert_size_stride(primals_92, (960, ), (1, ))
    assert_size_stride(primals_93, (1000, 1280), (1280, 1))
    assert_size_stride(primals_94, (1000, ), (1, ))
    assert_size_stride(primals_95, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_96, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_97, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_98, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_99, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_100, (24, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_101, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_102, (72, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_103, (24, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_104, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_105, (72, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_106, (24, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_107, (24, ), (1, ))
    assert_size_stride(primals_108, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_109, (72, ), (1, ))
    assert_size_stride(primals_110, (40, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_111, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_112, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_113, (32, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_114, (32, ), (1, ))
    assert_size_stride(primals_115, (120, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_116, (120, ), (1, ))
    assert_size_stride(primals_117, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_118, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_119, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_120, (32, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_121, (32, ), (1, ))
    assert_size_stride(primals_122, (120, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_123, (120, ), (1, ))
    assert_size_stride(primals_124, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_125, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_126, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_127, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_128, (200, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_129, (200, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_130, (80, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(primals_131, (184, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_132, (184, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_133, (80, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_134, (184, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_135, (184, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_136, (80, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_137, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_138, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_139, (120, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_140, (120, ), (1, ))
    assert_size_stride(primals_141, (480, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_142, (480, ), (1, ))
    assert_size_stride(primals_143, (112, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_144, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_145, (672, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_146, (168, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_147, (168, ), (1, ))
    assert_size_stride(primals_148, (672, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_149, (672, ), (1, ))
    assert_size_stride(primals_150, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_151, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_152, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_153, (168, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_154, (168, ), (1, ))
    assert_size_stride(primals_155, (672, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_156, (672, ), (1, ))
    assert_size_stride(primals_157, (160, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_158, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_159, (960, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_160, (240, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_161, (240, ), (1, ))
    assert_size_stride(primals_162, (960, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_163, (960, ), (1, ))
    assert_size_stride(primals_164, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_165, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_166, (960, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_167, (240, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_168, (240, ), (1, ))
    assert_size_stride(primals_169, (960, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_170, (960, ), (1, ))
    assert_size_stride(primals_171, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_172, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_173, (1280, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_174, (1280, ), (1, ))
    assert_size_stride(primals_175, (), ())
    assert_size_stride(primals_176, (16, ), (1, ))
    assert_size_stride(primals_177, (16, ), (1, ))
    assert_size_stride(primals_178, (), ())
    assert_size_stride(primals_179, (16, ), (1, ))
    assert_size_stride(primals_180, (16, ), (1, ))
    assert_size_stride(primals_181, (), ())
    assert_size_stride(primals_182, (16, ), (1, ))
    assert_size_stride(primals_183, (16, ), (1, ))
    assert_size_stride(primals_184, (), ())
    assert_size_stride(primals_185, (64, ), (1, ))
    assert_size_stride(primals_186, (64, ), (1, ))
    assert_size_stride(primals_187, (), ())
    assert_size_stride(primals_188, (64, ), (1, ))
    assert_size_stride(primals_189, (64, ), (1, ))
    assert_size_stride(primals_190, (), ())
    assert_size_stride(primals_191, (24, ), (1, ))
    assert_size_stride(primals_192, (24, ), (1, ))
    assert_size_stride(primals_193, (), ())
    assert_size_stride(primals_194, (72, ), (1, ))
    assert_size_stride(primals_195, (72, ), (1, ))
    assert_size_stride(primals_196, (), ())
    assert_size_stride(primals_197, (72, ), (1, ))
    assert_size_stride(primals_198, (72, ), (1, ))
    assert_size_stride(primals_199, (), ())
    assert_size_stride(primals_200, (24, ), (1, ))
    assert_size_stride(primals_201, (24, ), (1, ))
    assert_size_stride(primals_202, (), ())
    assert_size_stride(primals_203, (72, ), (1, ))
    assert_size_stride(primals_204, (72, ), (1, ))
    assert_size_stride(primals_205, (), ())
    assert_size_stride(primals_206, (72, ), (1, ))
    assert_size_stride(primals_207, (72, ), (1, ))
    assert_size_stride(primals_208, (), ())
    assert_size_stride(primals_209, (40, ), (1, ))
    assert_size_stride(primals_210, (40, ), (1, ))
    assert_size_stride(primals_211, (), ())
    assert_size_stride(primals_212, (120, ), (1, ))
    assert_size_stride(primals_213, (120, ), (1, ))
    assert_size_stride(primals_214, (), ())
    assert_size_stride(primals_215, (120, ), (1, ))
    assert_size_stride(primals_216, (120, ), (1, ))
    assert_size_stride(primals_217, (), ())
    assert_size_stride(primals_218, (40, ), (1, ))
    assert_size_stride(primals_219, (40, ), (1, ))
    assert_size_stride(primals_220, (), ())
    assert_size_stride(primals_221, (120, ), (1, ))
    assert_size_stride(primals_222, (120, ), (1, ))
    assert_size_stride(primals_223, (), ())
    assert_size_stride(primals_224, (120, ), (1, ))
    assert_size_stride(primals_225, (120, ), (1, ))
    assert_size_stride(primals_226, (), ())
    assert_size_stride(primals_227, (40, ), (1, ))
    assert_size_stride(primals_228, (40, ), (1, ))
    assert_size_stride(primals_229, (), ())
    assert_size_stride(primals_230, (240, ), (1, ))
    assert_size_stride(primals_231, (240, ), (1, ))
    assert_size_stride(primals_232, (), ())
    assert_size_stride(primals_233, (240, ), (1, ))
    assert_size_stride(primals_234, (240, ), (1, ))
    assert_size_stride(primals_235, (), ())
    assert_size_stride(primals_236, (80, ), (1, ))
    assert_size_stride(primals_237, (80, ), (1, ))
    assert_size_stride(primals_238, (), ())
    assert_size_stride(primals_239, (200, ), (1, ))
    assert_size_stride(primals_240, (200, ), (1, ))
    assert_size_stride(primals_241, (), ())
    assert_size_stride(primals_242, (200, ), (1, ))
    assert_size_stride(primals_243, (200, ), (1, ))
    assert_size_stride(primals_244, (), ())
    assert_size_stride(primals_245, (80, ), (1, ))
    assert_size_stride(primals_246, (80, ), (1, ))
    assert_size_stride(primals_247, (), ())
    assert_size_stride(primals_248, (184, ), (1, ))
    assert_size_stride(primals_249, (184, ), (1, ))
    assert_size_stride(primals_250, (), ())
    assert_size_stride(primals_251, (184, ), (1, ))
    assert_size_stride(primals_252, (184, ), (1, ))
    assert_size_stride(primals_253, (), ())
    assert_size_stride(primals_254, (80, ), (1, ))
    assert_size_stride(primals_255, (80, ), (1, ))
    assert_size_stride(primals_256, (), ())
    assert_size_stride(primals_257, (184, ), (1, ))
    assert_size_stride(primals_258, (184, ), (1, ))
    assert_size_stride(primals_259, (), ())
    assert_size_stride(primals_260, (184, ), (1, ))
    assert_size_stride(primals_261, (184, ), (1, ))
    assert_size_stride(primals_262, (), ())
    assert_size_stride(primals_263, (80, ), (1, ))
    assert_size_stride(primals_264, (80, ), (1, ))
    assert_size_stride(primals_265, (), ())
    assert_size_stride(primals_266, (480, ), (1, ))
    assert_size_stride(primals_267, (480, ), (1, ))
    assert_size_stride(primals_268, (), ())
    assert_size_stride(primals_269, (480, ), (1, ))
    assert_size_stride(primals_270, (480, ), (1, ))
    assert_size_stride(primals_271, (), ())
    assert_size_stride(primals_272, (112, ), (1, ))
    assert_size_stride(primals_273, (112, ), (1, ))
    assert_size_stride(primals_274, (), ())
    assert_size_stride(primals_275, (672, ), (1, ))
    assert_size_stride(primals_276, (672, ), (1, ))
    assert_size_stride(primals_277, (), ())
    assert_size_stride(primals_278, (672, ), (1, ))
    assert_size_stride(primals_279, (672, ), (1, ))
    assert_size_stride(primals_280, (), ())
    assert_size_stride(primals_281, (112, ), (1, ))
    assert_size_stride(primals_282, (112, ), (1, ))
    assert_size_stride(primals_283, (), ())
    assert_size_stride(primals_284, (672, ), (1, ))
    assert_size_stride(primals_285, (672, ), (1, ))
    assert_size_stride(primals_286, (), ())
    assert_size_stride(primals_287, (672, ), (1, ))
    assert_size_stride(primals_288, (672, ), (1, ))
    assert_size_stride(primals_289, (), ())
    assert_size_stride(primals_290, (160, ), (1, ))
    assert_size_stride(primals_291, (160, ), (1, ))
    assert_size_stride(primals_292, (), ())
    assert_size_stride(primals_293, (960, ), (1, ))
    assert_size_stride(primals_294, (960, ), (1, ))
    assert_size_stride(primals_295, (), ())
    assert_size_stride(primals_296, (960, ), (1, ))
    assert_size_stride(primals_297, (960, ), (1, ))
    assert_size_stride(primals_298, (), ())
    assert_size_stride(primals_299, (160, ), (1, ))
    assert_size_stride(primals_300, (160, ), (1, ))
    assert_size_stride(primals_301, (), ())
    assert_size_stride(primals_302, (960, ), (1, ))
    assert_size_stride(primals_303, (960, ), (1, ))
    assert_size_stride(primals_304, (), ())
    assert_size_stride(primals_305, (960, ), (1, ))
    assert_size_stride(primals_306, (960, ), (1, ))
    assert_size_stride(primals_307, (), ())
    assert_size_stride(primals_308, (160, ), (1, ))
    assert_size_stride(primals_309, (160, ), (1, ))
    assert_size_stride(primals_310, (), ())
    assert_size_stride(primals_311, (960, ), (1, ))
    assert_size_stride(primals_312, (960, ), (1, ))
    assert_size_stride(primals_313, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((16, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    cpp_fused_0(c_void_p(primals_95.data_ptr()), c_void_p(primals_313.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del primals_313
    del primals_95
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf1, buf0, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf2, (8, 16, 112, 112), (200704, 1, 1792, 16))
    buf3 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cpu', dtype=torch.float32)
    buf6 = empty((16, ), device='cpu', dtype=torch.float32)
    buf7 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    buf8 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_hardswish_1(c_void_p(buf2.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()))
    del primals_2
    # Source Nodes: [x_5], Original ATen: [aten.convolution]
    buf9 = extern_kernels.convolution(buf8, primals_96, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
    assert_size_stride(buf9, (8, 16, 112, 112), (200704, 1, 1792, 16))
    buf10 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cpu', dtype=torch.float32)
    buf11 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cpu', dtype=torch.float32)
    buf13 = empty((16, ), device='cpu', dtype=torch.float32)
    buf14 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_2(c_void_p(buf9.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()))
    del primals_4
    # Source Nodes: [x_11], Original ATen: [aten.convolution]
    buf15 = extern_kernels.convolution(buf14, primals_97, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf15, (8, 16, 112, 112), (200704, 1, 1792, 16))
    buf16 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cpu', dtype=torch.float32)
    buf17 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cpu', dtype=torch.float32)
    buf19 = empty((16, ), device='cpu', dtype=torch.float32)
    buf20 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_3(c_void_p(buf15.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()))
    del primals_6
    # Source Nodes: [x_17], Original ATen: [aten.convolution]
    buf21 = extern_kernels.convolution(buf20, primals_98, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf21, (8, 64, 112, 112), (802816, 1, 7168, 64))
    buf22 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf23 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf25 = empty((64, ), device='cpu', dtype=torch.float32)
    buf26 = empty_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_4(c_void_p(buf21.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()))
    del primals_8
    # Source Nodes: [x_22], Original ATen: [aten.convolution]
    buf27 = extern_kernels.convolution(buf26, primals_99, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
    assert_size_stride(buf27, (8, 64, 56, 56), (200704, 1, 3584, 64))
    buf28 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf29 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf31 = empty((64, ), device='cpu', dtype=torch.float32)
    buf32 = empty_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_5(c_void_p(buf27.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf32.data_ptr()))
    del primals_10
    # Source Nodes: [x_28], Original ATen: [aten.convolution]
    buf33 = extern_kernels.convolution(buf32, primals_100, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf33, (8, 24, 56, 56), (75264, 1, 1344, 24))
    buf34 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cpu', dtype=torch.float32)
    buf35 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cpu', dtype=torch.float32)
    buf37 = empty((24, ), device='cpu', dtype=torch.float32)
    buf38 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_6(c_void_p(buf33.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf38.data_ptr()))
    del primals_12
    # Source Nodes: [x_33], Original ATen: [aten.convolution]
    buf39 = extern_kernels.convolution(buf38, primals_101, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf39, (8, 72, 56, 56), (225792, 1, 4032, 72))
    buf40 = empty_strided((1, 72, 1, 1), (72, 1, 72, 72), device='cpu', dtype=torch.float32)
    buf41 = empty_strided((1, 72, 1, 1), (72, 1, 72, 72), device='cpu', dtype=torch.float32)
    buf43 = empty((72, ), device='cpu', dtype=torch.float32)
    buf44 = empty_strided((8, 72, 56, 56), (225792, 1, 4032, 72), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_7(c_void_p(buf39.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf44.data_ptr()))
    del primals_14
    # Source Nodes: [x_38], Original ATen: [aten.convolution]
    buf45 = extern_kernels.convolution(buf44, primals_102, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=72, bias=None)
    assert_size_stride(buf45, (8, 72, 56, 56), (225792, 1, 4032, 72))
    buf46 = empty_strided((1, 72, 1, 1), (72, 1, 72, 72), device='cpu', dtype=torch.float32)
    buf47 = empty_strided((1, 72, 1, 1), (72, 1, 72, 72), device='cpu', dtype=torch.float32)
    buf49 = empty((72, ), device='cpu', dtype=torch.float32)
    buf50 = empty_strided((8, 72, 56, 56), (225792, 1, 4032, 72), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_8(c_void_p(buf45.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf50.data_ptr()))
    del primals_16
    # Source Nodes: [x_44], Original ATen: [aten.convolution]
    buf51 = extern_kernels.convolution(buf50, primals_103, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf51, (8, 24, 56, 56), (75264, 1, 1344, 24))
    buf52 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cpu', dtype=torch.float32)
    buf53 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cpu', dtype=torch.float32)
    buf55 = empty((24, ), device='cpu', dtype=torch.float32)
    buf56 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_9(c_void_p(buf51.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()))
    del primals_18
    # Source Nodes: [x_50], Original ATen: [aten.convolution]
    buf57 = extern_kernels.convolution(buf56, primals_104, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf57, (8, 72, 56, 56), (225792, 1, 4032, 72))
    buf58 = empty_strided((1, 72, 1, 1), (72, 1, 72, 72), device='cpu', dtype=torch.float32)
    buf59 = empty_strided((1, 72, 1, 1), (72, 1, 72, 72), device='cpu', dtype=torch.float32)
    buf61 = empty((72, ), device='cpu', dtype=torch.float32)
    buf62 = empty_strided((8, 72, 56, 56), (225792, 1, 4032, 72), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_10(c_void_p(buf57.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()))
    del primals_20
    # Source Nodes: [x_55], Original ATen: [aten.convolution]
    buf63 = extern_kernels.convolution(buf62, primals_105, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=72, bias=None)
    assert_size_stride(buf63, (8, 72, 28, 28), (56448, 1, 2016, 72))
    buf64 = empty_strided((1, 72, 1, 1), (72, 1, 72, 72), device='cpu', dtype=torch.float32)
    buf65 = empty_strided((1, 72, 1, 1), (72, 1, 72, 72), device='cpu', dtype=torch.float32)
    buf67 = empty((72, ), device='cpu', dtype=torch.float32)
    buf68 = empty_strided((8, 72, 28, 28), (56448, 1, 2016, 72), device='cpu', dtype=torch.float32)
    buf69 = empty_strided((8, 72, 1, 1), (72, 1, 576, 576), device='cpu', dtype=torch.float32)
    buf70 = reinterpret_tensor(buf69, (8, 72, 1, 1), (72, 1, 72, 72), 0); del buf69  # reuse
    cpp_fused__native_batch_norm_legit_functional_mean_relu_11(c_void_p(buf70.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf68.data_ptr()))
    del primals_22
    # Source Nodes: [x_se_1], Original ATen: [aten.convolution]
    buf71 = extern_kernels.convolution(buf70, primals_106, primals_107, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf71, (8, 24, 1, 1), (24, 1, 24, 24))
    del primals_107
    buf72 = buf71; del buf71  # reuse
    cpp_fused_relu_12(c_void_p(buf72.data_ptr()))
    # Source Nodes: [x_se_3], Original ATen: [aten.convolution]
    buf73 = extern_kernels.convolution(buf72, primals_108, primals_109, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf73, (8, 72, 1, 1), (72, 1, 72, 72))
    del primals_109
    buf74 = empty_strided((8, 72, 1, 1), (72, 1, 72, 72), device='cpu', dtype=torch.float32)
    buf75 = empty_strided((8, 72, 28, 28), (56448, 1, 2016, 72), device='cpu', dtype=torch.float32)
    cpp_fused_hardsigmoid_mul_13(c_void_p(buf73.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf75.data_ptr()))
    # Source Nodes: [x_61], Original ATen: [aten.convolution]
    buf76 = extern_kernels.convolution(buf75, primals_110, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf76, (8, 40, 28, 28), (31360, 1, 1120, 40))
    buf77 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cpu', dtype=torch.float32)
    buf78 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cpu', dtype=torch.float32)
    buf80 = empty((40, ), device='cpu', dtype=torch.float32)
    buf81 = empty_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_14(c_void_p(buf76.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf81.data_ptr()))
    del primals_24
    # Source Nodes: [x_66], Original ATen: [aten.convolution]
    buf82 = extern_kernels.convolution(buf81, primals_111, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf82, (8, 120, 28, 28), (94080, 1, 3360, 120))
    buf83 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.float32)
    buf84 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.float32)
    buf86 = empty((120, ), device='cpu', dtype=torch.float32)
    buf87 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_15(c_void_p(buf82.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()))
    del primals_26
    # Source Nodes: [x_71], Original ATen: [aten.convolution]
    buf88 = extern_kernels.convolution(buf87, primals_112, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
    assert_size_stride(buf88, (8, 120, 28, 28), (94080, 1, 3360, 120))
    buf89 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.float32)
    buf90 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.float32)
    buf92 = empty((120, ), device='cpu', dtype=torch.float32)
    buf93 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    buf94 = empty_strided((8, 120, 1, 1), (120, 1, 960, 960), device='cpu', dtype=torch.float32)
    buf95 = reinterpret_tensor(buf94, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf94  # reuse
    cpp_fused__native_batch_norm_legit_functional_mean_relu_16(c_void_p(buf95.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf93.data_ptr()))
    del primals_28
    # Source Nodes: [x_se_5], Original ATen: [aten.convolution]
    buf96 = extern_kernels.convolution(buf95, primals_113, primals_114, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf96, (8, 32, 1, 1), (32, 1, 32, 32))
    del primals_114
    buf97 = buf96; del buf96  # reuse
    cpp_fused_relu_17(c_void_p(buf97.data_ptr()))
    # Source Nodes: [x_se_7], Original ATen: [aten.convolution]
    buf98 = extern_kernels.convolution(buf97, primals_115, primals_116, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf98, (8, 120, 1, 1), (120, 1, 120, 120))
    del primals_116
    buf99 = empty_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.float32)
    buf100 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    cpp_fused_hardsigmoid_mul_18(c_void_p(buf98.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf100.data_ptr()))
    # Source Nodes: [x_77], Original ATen: [aten.convolution]
    buf101 = extern_kernels.convolution(buf100, primals_117, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf101, (8, 40, 28, 28), (31360, 1, 1120, 40))
    buf102 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cpu', dtype=torch.float32)
    buf103 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cpu', dtype=torch.float32)
    buf105 = empty((40, ), device='cpu', dtype=torch.float32)
    buf106 = empty_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_19(c_void_p(buf101.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()))
    del primals_30
    # Source Nodes: [x_83], Original ATen: [aten.convolution]
    buf107 = extern_kernels.convolution(buf106, primals_118, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf107, (8, 120, 28, 28), (94080, 1, 3360, 120))
    buf108 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.float32)
    buf109 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.float32)
    buf111 = empty((120, ), device='cpu', dtype=torch.float32)
    buf112 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_20(c_void_p(buf107.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf112.data_ptr()))
    del primals_32
    # Source Nodes: [x_88], Original ATen: [aten.convolution]
    buf113 = extern_kernels.convolution(buf112, primals_119, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
    assert_size_stride(buf113, (8, 120, 28, 28), (94080, 1, 3360, 120))
    buf114 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.float32)
    buf115 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.float32)
    buf117 = empty((120, ), device='cpu', dtype=torch.float32)
    buf118 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    buf119 = empty_strided((8, 120, 1, 1), (120, 1, 960, 960), device='cpu', dtype=torch.float32)
    buf120 = reinterpret_tensor(buf119, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf119  # reuse
    cpp_fused__native_batch_norm_legit_functional_mean_relu_21(c_void_p(buf120.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(primals_34.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf118.data_ptr()))
    del primals_34
    # Source Nodes: [x_se_9], Original ATen: [aten.convolution]
    buf121 = extern_kernels.convolution(buf120, primals_120, primals_121, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf121, (8, 32, 1, 1), (32, 1, 32, 32))
    del primals_121
    buf122 = buf121; del buf121  # reuse
    cpp_fused_relu_22(c_void_p(buf122.data_ptr()))
    # Source Nodes: [x_se_11], Original ATen: [aten.convolution]
    buf123 = extern_kernels.convolution(buf122, primals_122, primals_123, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf123, (8, 120, 1, 1), (120, 1, 120, 120))
    del primals_123
    buf124 = empty_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.float32)
    buf125 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    cpp_fused_hardsigmoid_mul_23(c_void_p(buf123.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf125.data_ptr()))
    # Source Nodes: [x_94], Original ATen: [aten.convolution]
    buf126 = extern_kernels.convolution(buf125, primals_124, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf126, (8, 40, 28, 28), (31360, 1, 1120, 40))
    buf127 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cpu', dtype=torch.float32)
    buf128 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cpu', dtype=torch.float32)
    buf130 = empty((40, ), device='cpu', dtype=torch.float32)
    buf131 = empty_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_24(c_void_p(buf126.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf131.data_ptr()))
    del primals_36
    # Source Nodes: [x_100], Original ATen: [aten.convolution]
    buf132 = extern_kernels.convolution(buf131, primals_125, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf132, (8, 240, 28, 28), (188160, 1, 6720, 240))
    buf133 = empty_strided((1, 240, 1, 1), (240, 1, 240, 240), device='cpu', dtype=torch.float32)
    buf134 = empty_strided((1, 240, 1, 1), (240, 1, 240, 240), device='cpu', dtype=torch.float32)
    buf136 = empty((240, ), device='cpu', dtype=torch.float32)
    buf137 = empty_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cpu', dtype=torch.float32)
    buf138 = empty_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_hardswish_25(c_void_p(buf132.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf138.data_ptr()))
    del primals_38
    # Source Nodes: [x_105], Original ATen: [aten.convolution]
    buf139 = extern_kernels.convolution(buf138, primals_126, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
    assert_size_stride(buf139, (8, 240, 14, 14), (47040, 1, 3360, 240))
    buf140 = empty_strided((1, 240, 1, 1), (240, 1, 240, 240), device='cpu', dtype=torch.float32)
    buf141 = empty_strided((1, 240, 1, 1), (240, 1, 240, 240), device='cpu', dtype=torch.float32)
    buf143 = empty((240, ), device='cpu', dtype=torch.float32)
    buf144 = empty_strided((8, 240, 14, 14), (47040, 1, 3360, 240), device='cpu', dtype=torch.float32)
    buf145 = empty_strided((8, 240, 14, 14), (47040, 1, 3360, 240), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_hardswish_26(c_void_p(buf139.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(primals_40.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()))
    del primals_40
    # Source Nodes: [x_111], Original ATen: [aten.convolution]
    buf146 = extern_kernels.convolution(buf145, primals_127, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf146, (8, 80, 14, 14), (15680, 1, 1120, 80))
    buf147 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    buf148 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    buf150 = empty((80, ), device='cpu', dtype=torch.float32)
    buf151 = empty_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_27(c_void_p(buf146.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf151.data_ptr()))
    del primals_42
    # Source Nodes: [x_116], Original ATen: [aten.convolution]
    buf152 = extern_kernels.convolution(buf151, primals_128, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf152, (8, 200, 14, 14), (39200, 1, 2800, 200))
    buf153 = empty_strided((1, 200, 1, 1), (200, 1, 200, 200), device='cpu', dtype=torch.float32)
    buf154 = empty_strided((1, 200, 1, 1), (200, 1, 200, 200), device='cpu', dtype=torch.float32)
    buf156 = empty((200, ), device='cpu', dtype=torch.float32)
    buf157 = empty_strided((8, 200, 14, 14), (39200, 1, 2800, 200), device='cpu', dtype=torch.float32)
    buf158 = empty_strided((8, 200, 14, 14), (39200, 1, 2800, 200), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_hardswish_28(c_void_p(buf152.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(primals_44.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf158.data_ptr()))
    del primals_44
    # Source Nodes: [x_121], Original ATen: [aten.convolution]
    buf159 = extern_kernels.convolution(buf158, primals_129, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=200, bias=None)
    assert_size_stride(buf159, (8, 200, 14, 14), (39200, 1, 2800, 200))
    buf160 = empty_strided((1, 200, 1, 1), (200, 1, 200, 200), device='cpu', dtype=torch.float32)
    buf161 = empty_strided((1, 200, 1, 1), (200, 1, 200, 200), device='cpu', dtype=torch.float32)
    buf163 = empty((200, ), device='cpu', dtype=torch.float32)
    buf164 = empty_strided((8, 200, 14, 14), (39200, 1, 2800, 200), device='cpu', dtype=torch.float32)
    buf165 = empty_strided((8, 200, 14, 14), (39200, 1, 2800, 200), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_hardswish_29(c_void_p(buf159.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(primals_46.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf165.data_ptr()))
    del primals_46
    # Source Nodes: [x_127], Original ATen: [aten.convolution]
    buf166 = extern_kernels.convolution(buf165, primals_130, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf166, (8, 80, 14, 14), (15680, 1, 1120, 80))
    buf167 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    buf168 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    buf170 = empty((80, ), device='cpu', dtype=torch.float32)
    buf171 = empty_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_30(c_void_p(buf166.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(primals_48.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf171.data_ptr()))
    del primals_48
    # Source Nodes: [x_133], Original ATen: [aten.convolution]
    buf172 = extern_kernels.convolution(buf171, primals_131, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf172, (8, 184, 14, 14), (36064, 1, 2576, 184))
    buf173 = empty_strided((1, 184, 1, 1), (184, 1, 184, 184), device='cpu', dtype=torch.float32)
    buf174 = empty_strided((1, 184, 1, 1), (184, 1, 184, 184), device='cpu', dtype=torch.float32)
    buf176 = empty((184, ), device='cpu', dtype=torch.float32)
    buf177 = empty_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cpu', dtype=torch.float32)
    buf178 = empty_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_hardswish_31(c_void_p(buf172.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf178.data_ptr()))
    del primals_50
    # Source Nodes: [x_138], Original ATen: [aten.convolution]
    buf179 = extern_kernels.convolution(buf178, primals_132, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=184, bias=None)
    assert_size_stride(buf179, (8, 184, 14, 14), (36064, 1, 2576, 184))
    buf180 = empty_strided((1, 184, 1, 1), (184, 1, 184, 184), device='cpu', dtype=torch.float32)
    buf181 = empty_strided((1, 184, 1, 1), (184, 1, 184, 184), device='cpu', dtype=torch.float32)
    buf183 = empty((184, ), device='cpu', dtype=torch.float32)
    buf184 = empty_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cpu', dtype=torch.float32)
    buf185 = empty_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_hardswish_32(c_void_p(buf179.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(primals_52.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()))
    del primals_52
    # Source Nodes: [x_144], Original ATen: [aten.convolution]
    buf186 = extern_kernels.convolution(buf185, primals_133, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf186, (8, 80, 14, 14), (15680, 1, 1120, 80))
    buf187 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    buf188 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    buf190 = empty((80, ), device='cpu', dtype=torch.float32)
    buf191 = empty_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_33(c_void_p(buf186.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf191.data_ptr()))
    del primals_54
    # Source Nodes: [x_150], Original ATen: [aten.convolution]
    buf192 = extern_kernels.convolution(buf191, primals_134, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf192, (8, 184, 14, 14), (36064, 1, 2576, 184))
    buf193 = empty_strided((1, 184, 1, 1), (184, 1, 184, 184), device='cpu', dtype=torch.float32)
    buf194 = empty_strided((1, 184, 1, 1), (184, 1, 184, 184), device='cpu', dtype=torch.float32)
    buf196 = empty((184, ), device='cpu', dtype=torch.float32)
    buf197 = empty_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cpu', dtype=torch.float32)
    buf198 = empty_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_hardswish_34(c_void_p(buf192.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(primals_56.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf198.data_ptr()))
    del primals_56
    # Source Nodes: [x_155], Original ATen: [aten.convolution]
    buf199 = extern_kernels.convolution(buf198, primals_135, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=184, bias=None)
    assert_size_stride(buf199, (8, 184, 14, 14), (36064, 1, 2576, 184))
    buf200 = empty_strided((1, 184, 1, 1), (184, 1, 184, 184), device='cpu', dtype=torch.float32)
    buf201 = empty_strided((1, 184, 1, 1), (184, 1, 184, 184), device='cpu', dtype=torch.float32)
    buf203 = empty((184, ), device='cpu', dtype=torch.float32)
    buf204 = empty_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cpu', dtype=torch.float32)
    buf205 = empty_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_hardswish_35(c_void_p(buf199.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(primals_58.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf205.data_ptr()))
    del primals_58
    # Source Nodes: [x_161], Original ATen: [aten.convolution]
    buf206 = extern_kernels.convolution(buf205, primals_136, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf206, (8, 80, 14, 14), (15680, 1, 1120, 80))
    buf207 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    buf208 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    buf210 = empty((80, ), device='cpu', dtype=torch.float32)
    buf211 = empty_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_36(c_void_p(buf206.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(primals_60.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf211.data_ptr()))
    del primals_60
    # Source Nodes: [x_167], Original ATen: [aten.convolution]
    buf212 = extern_kernels.convolution(buf211, primals_137, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf212, (8, 480, 14, 14), (94080, 1, 6720, 480))
    buf213 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    buf214 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    buf216 = empty((480, ), device='cpu', dtype=torch.float32)
    buf217 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    buf218 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_hardswish_37(c_void_p(buf212.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf218.data_ptr()))
    del primals_62
    # Source Nodes: [x_172], Original ATen: [aten.convolution]
    buf219 = extern_kernels.convolution(buf218, primals_138, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
    assert_size_stride(buf219, (8, 480, 14, 14), (94080, 1, 6720, 480))
    buf220 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    buf221 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    buf223 = empty((480, ), device='cpu', dtype=torch.float32)
    buf224 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    buf225 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    buf226 = empty_strided((8, 480, 1, 1), (480, 1, 3840, 3840), device='cpu', dtype=torch.float32)
    buf227 = reinterpret_tensor(buf226, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf226  # reuse
    cpp_fused__native_batch_norm_legit_functional_hardswish_mean_38(c_void_p(buf227.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(primals_64.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf225.data_ptr()))
    del primals_64
    # Source Nodes: [x_se_13], Original ATen: [aten.convolution]
    buf228 = extern_kernels.convolution(buf227, primals_139, primals_140, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf228, (8, 120, 1, 1), (120, 1, 120, 120))
    del primals_140
    buf229 = buf228; del buf228  # reuse
    cpp_fused_relu_39(c_void_p(buf229.data_ptr()))
    # Source Nodes: [x_se_15], Original ATen: [aten.convolution]
    buf230 = extern_kernels.convolution(buf229, primals_141, primals_142, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf230, (8, 480, 1, 1), (480, 1, 480, 480))
    del primals_142
    buf231 = empty_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    buf232 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    cpp_fused_hardsigmoid_mul_40(c_void_p(buf230.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf232.data_ptr()))
    # Source Nodes: [x_178], Original ATen: [aten.convolution]
    buf233 = extern_kernels.convolution(buf232, primals_143, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf233, (8, 112, 14, 14), (21952, 1, 1568, 112))
    buf234 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cpu', dtype=torch.float32)
    buf235 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cpu', dtype=torch.float32)
    buf237 = empty((112, ), device='cpu', dtype=torch.float32)
    buf238 = empty_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_41(c_void_p(buf233.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(primals_66.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()))
    del primals_66
    # Source Nodes: [x_183], Original ATen: [aten.convolution]
    buf239 = extern_kernels.convolution(buf238, primals_144, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf239, (8, 672, 14, 14), (131712, 1, 9408, 672))
    buf240 = empty_strided((1, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    buf241 = empty_strided((1, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    buf243 = empty((672, ), device='cpu', dtype=torch.float32)
    buf244 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    buf245 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_hardswish_42(c_void_p(buf239.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf245.data_ptr()))
    del primals_68
    # Source Nodes: [x_188], Original ATen: [aten.convolution]
    buf246 = extern_kernels.convolution(buf245, primals_145, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
    assert_size_stride(buf246, (8, 672, 14, 14), (131712, 1, 9408, 672))
    buf247 = empty_strided((1, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    buf248 = empty_strided((1, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    buf250 = empty((672, ), device='cpu', dtype=torch.float32)
    buf251 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    buf252 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    buf253 = empty_strided((8, 672, 1, 1), (672, 1, 5376, 5376), device='cpu', dtype=torch.float32)
    buf254 = reinterpret_tensor(buf253, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf253  # reuse
    cpp_fused__native_batch_norm_legit_functional_hardswish_mean_43(c_void_p(buf254.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(primals_70.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf252.data_ptr()))
    del primals_70
    # Source Nodes: [x_se_17], Original ATen: [aten.convolution]
    buf255 = extern_kernels.convolution(buf254, primals_146, primals_147, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf255, (8, 168, 1, 1), (168, 1, 168, 168))
    del primals_147
    buf256 = buf255; del buf255  # reuse
    cpp_fused_relu_44(c_void_p(buf256.data_ptr()))
    # Source Nodes: [x_se_19], Original ATen: [aten.convolution]
    buf257 = extern_kernels.convolution(buf256, primals_148, primals_149, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf257, (8, 672, 1, 1), (672, 1, 672, 672))
    del primals_149
    buf258 = empty_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    buf259 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    cpp_fused_hardsigmoid_mul_45(c_void_p(buf257.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf259.data_ptr()))
    # Source Nodes: [x_194], Original ATen: [aten.convolution]
    buf260 = extern_kernels.convolution(buf259, primals_150, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf260, (8, 112, 14, 14), (21952, 1, 1568, 112))
    buf261 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cpu', dtype=torch.float32)
    buf262 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cpu', dtype=torch.float32)
    buf264 = empty((112, ), device='cpu', dtype=torch.float32)
    buf265 = empty_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_46(c_void_p(buf260.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(primals_72.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf265.data_ptr()))
    del primals_72
    # Source Nodes: [x_200], Original ATen: [aten.convolution]
    buf266 = extern_kernels.convolution(buf265, primals_151, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf266, (8, 672, 14, 14), (131712, 1, 9408, 672))
    buf267 = empty_strided((1, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    buf268 = empty_strided((1, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    buf270 = empty((672, ), device='cpu', dtype=torch.float32)
    buf271 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    buf272 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_hardswish_47(c_void_p(buf266.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf272.data_ptr()))
    del primals_74
    # Source Nodes: [x_205], Original ATen: [aten.convolution]
    buf273 = extern_kernels.convolution(buf272, primals_152, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
    assert_size_stride(buf273, (8, 672, 7, 7), (32928, 1, 4704, 672))
    buf274 = empty_strided((1, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    buf275 = empty_strided((1, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    buf277 = empty((672, ), device='cpu', dtype=torch.float32)
    buf278 = empty_strided((8, 672, 7, 7), (32928, 1, 4704, 672), device='cpu', dtype=torch.float32)
    buf279 = empty_strided((8, 672, 7, 7), (32928, 1, 4704, 672), device='cpu', dtype=torch.float32)
    buf280 = empty_strided((8, 672, 1, 1), (672, 1, 5376, 5376), device='cpu', dtype=torch.float32)
    buf281 = reinterpret_tensor(buf280, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf280  # reuse
    cpp_fused__native_batch_norm_legit_functional_hardswish_mean_48(c_void_p(buf281.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(primals_76.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf279.data_ptr()))
    del primals_76
    # Source Nodes: [x_se_21], Original ATen: [aten.convolution]
    buf282 = extern_kernels.convolution(buf281, primals_153, primals_154, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf282, (8, 168, 1, 1), (168, 1, 168, 168))
    del primals_154
    buf283 = buf282; del buf282  # reuse
    cpp_fused_relu_49(c_void_p(buf283.data_ptr()))
    # Source Nodes: [x_se_23], Original ATen: [aten.convolution]
    buf284 = extern_kernels.convolution(buf283, primals_155, primals_156, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf284, (8, 672, 1, 1), (672, 1, 672, 672))
    del primals_156
    buf285 = empty_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    buf286 = empty_strided((8, 672, 7, 7), (32928, 1, 4704, 672), device='cpu', dtype=torch.float32)
    cpp_fused_hardsigmoid_mul_50(c_void_p(buf284.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf286.data_ptr()))
    # Source Nodes: [x_211], Original ATen: [aten.convolution]
    buf287 = extern_kernels.convolution(buf286, primals_157, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf287, (8, 160, 7, 7), (7840, 1, 1120, 160))
    buf288 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf289 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf291 = empty((160, ), device='cpu', dtype=torch.float32)
    buf292 = empty_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_51(c_void_p(buf287.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(primals_78.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf292.data_ptr()))
    del primals_78
    # Source Nodes: [x_216], Original ATen: [aten.convolution]
    buf293 = extern_kernels.convolution(buf292, primals_158, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf293, (8, 960, 7, 7), (47040, 1, 6720, 960))
    buf294 = empty_strided((1, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.float32)
    buf295 = empty_strided((1, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.float32)
    buf297 = empty((960, ), device='cpu', dtype=torch.float32)
    buf298 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    buf299 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_hardswish_52(c_void_p(buf293.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf299.data_ptr()))
    del primals_80
    # Source Nodes: [x_221], Original ATen: [aten.convolution]
    buf300 = extern_kernels.convolution(buf299, primals_159, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
    assert_size_stride(buf300, (8, 960, 7, 7), (47040, 1, 6720, 960))
    buf301 = empty_strided((1, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.float32)
    buf302 = empty_strided((1, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.float32)
    buf304 = empty((960, ), device='cpu', dtype=torch.float32)
    buf305 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    buf306 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    buf307 = empty_strided((8, 960, 1, 1), (960, 1, 7680, 7680), device='cpu', dtype=torch.float32)
    buf308 = reinterpret_tensor(buf307, (8, 960, 1, 1), (960, 1, 960, 960), 0); del buf307  # reuse
    cpp_fused__native_batch_norm_legit_functional_hardswish_mean_53(c_void_p(buf308.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(primals_82.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf306.data_ptr()))
    del primals_82
    # Source Nodes: [x_se_25], Original ATen: [aten.convolution]
    buf309 = extern_kernels.convolution(buf308, primals_160, primals_161, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf309, (8, 240, 1, 1), (240, 1, 240, 240))
    del primals_161
    buf310 = buf309; del buf309  # reuse
    cpp_fused_relu_54(c_void_p(buf310.data_ptr()))
    # Source Nodes: [x_se_27], Original ATen: [aten.convolution]
    buf311 = extern_kernels.convolution(buf310, primals_162, primals_163, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf311, (8, 960, 1, 1), (960, 1, 960, 960))
    del primals_163
    buf312 = empty_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.float32)
    buf359 = empty_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.bool)
    buf313 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    cpp_fused_hardsigmoid_hardsigmoid_backward_mul_55(c_void_p(buf311.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(buf313.data_ptr()))
    # Source Nodes: [x_227], Original ATen: [aten.convolution]
    buf314 = extern_kernels.convolution(buf313, primals_164, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf314, (8, 160, 7, 7), (7840, 1, 1120, 160))
    buf315 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf316 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf318 = empty((160, ), device='cpu', dtype=torch.float32)
    buf319 = empty_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_56(c_void_p(buf314.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(primals_84.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(buf319.data_ptr()))
    del primals_84
    # Source Nodes: [x_233], Original ATen: [aten.convolution]
    buf320 = extern_kernels.convolution(buf319, primals_165, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf320, (8, 960, 7, 7), (47040, 1, 6720, 960))
    buf321 = empty_strided((1, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.float32)
    buf322 = empty_strided((1, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.float32)
    buf324 = empty((960, ), device='cpu', dtype=torch.float32)
    buf325 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    buf326 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_hardswish_57(c_void_p(buf320.data_ptr()), c_void_p(primals_85.data_ptr()), c_void_p(primals_86.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf326.data_ptr()))
    del primals_86
    # Source Nodes: [x_238], Original ATen: [aten.convolution]
    buf327 = extern_kernels.convolution(buf326, primals_166, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
    assert_size_stride(buf327, (8, 960, 7, 7), (47040, 1, 6720, 960))
    buf328 = empty_strided((1, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.float32)
    buf329 = empty_strided((1, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.float32)
    buf331 = empty((960, ), device='cpu', dtype=torch.float32)
    buf332 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    buf333 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    buf334 = reinterpret_tensor(buf311, (8, 960, 1, 1), (960, 1, 7680, 7680), 0); del buf311  # reuse
    buf335 = reinterpret_tensor(buf334, (8, 960, 1, 1), (960, 1, 960, 960), 0); del buf334  # reuse
    cpp_fused__native_batch_norm_legit_functional_hardswish_mean_58(c_void_p(buf335.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(primals_88.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf333.data_ptr()))
    del primals_88
    # Source Nodes: [x_se_29], Original ATen: [aten.convolution]
    buf336 = extern_kernels.convolution(buf335, primals_167, primals_168, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf336, (8, 240, 1, 1), (240, 1, 240, 240))
    del primals_168
    buf337 = buf336; del buf336  # reuse
    cpp_fused_relu_59(c_void_p(buf337.data_ptr()))
    # Source Nodes: [x_se_31], Original ATen: [aten.convolution]
    buf338 = extern_kernels.convolution(buf337, primals_169, primals_170, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf338, (8, 960, 1, 1), (960, 1, 960, 960))
    del primals_170
    buf339 = empty_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.float32)
    buf358 = empty_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.bool)
    buf340 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    cpp_fused_hardsigmoid_hardsigmoid_backward_mul_60(c_void_p(buf338.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf340.data_ptr()))
    # Source Nodes: [x_244], Original ATen: [aten.convolution]
    buf341 = extern_kernels.convolution(buf340, primals_171, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf341, (8, 160, 7, 7), (7840, 1, 1120, 160))
    buf342 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf343 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf345 = empty((160, ), device='cpu', dtype=torch.float32)
    buf346 = empty_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_61(c_void_p(buf341.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(primals_90.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf346.data_ptr()))
    del primals_90
    # Source Nodes: [x_250], Original ATen: [aten.convolution]
    buf347 = extern_kernels.convolution(buf346, primals_172, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf347, (8, 960, 7, 7), (47040, 1, 6720, 960))
    buf348 = empty_strided((1, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.float32)
    buf349 = empty_strided((1, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.float32)
    buf351 = empty((960, ), device='cpu', dtype=torch.float32)
    buf352 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    buf353 = reinterpret_tensor(buf338, (8, 960, 1, 1), (960, 1, 7680, 7680), 0); del buf338  # reuse
    buf354 = reinterpret_tensor(buf353, (8, 960, 1, 1), (960, 1, 960, 960), 0); del buf353  # reuse
    cpp_fused__native_batch_norm_legit_functional_hardswish_mean_62(c_void_p(buf354.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(primals_92.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf352.data_ptr()))
    del primals_92
    # Source Nodes: [x_260], Original ATen: [aten.convolution]
    buf355 = extern_kernels.convolution(buf354, primals_173, primals_174, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf355, (8, 1280, 1, 1), (1280, 1, 1280, 1280))
    del primals_174
    buf356 = empty((8, 1280), device='cpu', dtype=torch.float32)
    cpp_fused_hardswish_view_63(c_void_p(buf355.data_ptr()), c_void_p(buf356.data_ptr()))
    buf357 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [pred], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_94, buf356, reinterpret_tensor(primals_93, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf357)
    del primals_94
    buf360 = empty_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.bool)
    buf361 = empty_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.bool)
    buf362 = empty_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.bool)
    buf363 = empty_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.bool)
    buf364 = empty_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.bool)
    buf365 = empty_strided((8, 72, 1, 1), (72, 1, 72, 72), device='cpu', dtype=torch.bool)
    buf372 = reinterpret_tensor(buf4, (16, ), (1, ), 0); del buf4  # reuse
    buf380 = reinterpret_tensor(buf11, (16, ), (1, ), 0); del buf11  # reuse
    buf388 = reinterpret_tensor(buf17, (16, ), (1, ), 0); del buf17  # reuse
    buf396 = reinterpret_tensor(buf23, (64, ), (1, ), 0); del buf23  # reuse
    buf404 = reinterpret_tensor(buf29, (64, ), (1, ), 0); del buf29  # reuse
    buf412 = reinterpret_tensor(buf35, (24, ), (1, ), 0); del buf35  # reuse
    buf420 = reinterpret_tensor(buf41, (72, ), (1, ), 0); del buf41  # reuse
    buf428 = reinterpret_tensor(buf47, (72, ), (1, ), 0); del buf47  # reuse
    buf436 = reinterpret_tensor(buf53, (24, ), (1, ), 0); del buf53  # reuse
    buf444 = reinterpret_tensor(buf59, (72, ), (1, ), 0); del buf59  # reuse
    buf452 = reinterpret_tensor(buf65, (72, ), (1, ), 0); del buf65  # reuse
    buf460 = reinterpret_tensor(buf78, (40, ), (1, ), 0); del buf78  # reuse
    buf468 = reinterpret_tensor(buf84, (120, ), (1, ), 0); del buf84  # reuse
    buf476 = reinterpret_tensor(buf90, (120, ), (1, ), 0); del buf90  # reuse
    buf484 = reinterpret_tensor(buf103, (40, ), (1, ), 0); del buf103  # reuse
    buf492 = reinterpret_tensor(buf109, (120, ), (1, ), 0); del buf109  # reuse
    buf500 = reinterpret_tensor(buf115, (120, ), (1, ), 0); del buf115  # reuse
    buf508 = reinterpret_tensor(buf128, (40, ), (1, ), 0); del buf128  # reuse
    buf516 = reinterpret_tensor(buf134, (240, ), (1, ), 0); del buf134  # reuse
    buf524 = reinterpret_tensor(buf141, (240, ), (1, ), 0); del buf141  # reuse
    buf532 = reinterpret_tensor(buf148, (80, ), (1, ), 0); del buf148  # reuse
    buf540 = reinterpret_tensor(buf154, (200, ), (1, ), 0); del buf154  # reuse
    buf548 = reinterpret_tensor(buf161, (200, ), (1, ), 0); del buf161  # reuse
    buf556 = reinterpret_tensor(buf168, (80, ), (1, ), 0); del buf168  # reuse
    buf564 = reinterpret_tensor(buf174, (184, ), (1, ), 0); del buf174  # reuse
    buf572 = reinterpret_tensor(buf181, (184, ), (1, ), 0); del buf181  # reuse
    buf580 = reinterpret_tensor(buf188, (80, ), (1, ), 0); del buf188  # reuse
    buf588 = reinterpret_tensor(buf194, (184, ), (1, ), 0); del buf194  # reuse
    buf596 = reinterpret_tensor(buf201, (184, ), (1, ), 0); del buf201  # reuse
    buf604 = reinterpret_tensor(buf208, (80, ), (1, ), 0); del buf208  # reuse
    buf612 = reinterpret_tensor(buf214, (480, ), (1, ), 0); del buf214  # reuse
    buf620 = reinterpret_tensor(buf221, (480, ), (1, ), 0); del buf221  # reuse
    buf628 = reinterpret_tensor(buf235, (112, ), (1, ), 0); del buf235  # reuse
    buf636 = reinterpret_tensor(buf241, (672, ), (1, ), 0); del buf241  # reuse
    buf644 = reinterpret_tensor(buf248, (672, ), (1, ), 0); del buf248  # reuse
    buf652 = reinterpret_tensor(buf262, (112, ), (1, ), 0); del buf262  # reuse
    buf660 = reinterpret_tensor(buf268, (672, ), (1, ), 0); del buf268  # reuse
    buf668 = reinterpret_tensor(buf275, (672, ), (1, ), 0); del buf275  # reuse
    buf676 = reinterpret_tensor(buf289, (160, ), (1, ), 0); del buf289  # reuse
    buf684 = reinterpret_tensor(buf295, (960, ), (1, ), 0); del buf295  # reuse
    buf692 = reinterpret_tensor(buf302, (960, ), (1, ), 0); del buf302  # reuse
    buf700 = reinterpret_tensor(buf316, (160, ), (1, ), 0); del buf316  # reuse
    buf708 = reinterpret_tensor(buf322, (960, ), (1, ), 0); del buf322  # reuse
    buf716 = reinterpret_tensor(buf329, (960, ), (1, ), 0); del buf329  # reuse
    buf724 = reinterpret_tensor(buf343, (160, ), (1, ), 0); del buf343  # reuse
    buf732 = reinterpret_tensor(buf349, (960, ), (1, ), 0); del buf349  # reuse
    cpp_fused__native_batch_norm_legit_functional_add_hardsigmoid_backward_64(c_void_p(buf372.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(buf484.data_ptr()), c_void_p(buf492.data_ptr()), c_void_p(buf500.data_ptr()), c_void_p(buf508.data_ptr()), c_void_p(buf516.data_ptr()), c_void_p(buf524.data_ptr()), c_void_p(buf532.data_ptr()), c_void_p(buf540.data_ptr()), c_void_p(buf548.data_ptr()), c_void_p(buf556.data_ptr()), c_void_p(buf564.data_ptr()), c_void_p(buf572.data_ptr()), c_void_p(buf580.data_ptr()), c_void_p(buf588.data_ptr()), c_void_p(buf596.data_ptr()), c_void_p(buf604.data_ptr()), c_void_p(buf612.data_ptr()), c_void_p(buf620.data_ptr()), c_void_p(buf628.data_ptr()), c_void_p(buf636.data_ptr()), c_void_p(buf644.data_ptr()), c_void_p(buf652.data_ptr()), c_void_p(buf660.data_ptr()), c_void_p(buf668.data_ptr()), c_void_p(buf676.data_ptr()), c_void_p(buf684.data_ptr()), c_void_p(buf692.data_ptr()), c_void_p(buf700.data_ptr()), c_void_p(buf708.data_ptr()), c_void_p(buf716.data_ptr()), c_void_p(buf724.data_ptr()), c_void_p(buf732.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(primals_175.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(primals_176.data_ptr()), c_void_p(primals_177.data_ptr()), c_void_p(primals_178.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(primals_179.data_ptr()), c_void_p(primals_180.data_ptr()), c_void_p(primals_181.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(primals_182.data_ptr()), c_void_p(primals_183.data_ptr()), c_void_p(primals_184.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(primals_185.data_ptr()), c_void_p(primals_186.data_ptr()), c_void_p(primals_187.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(primals_188.data_ptr()), c_void_p(primals_189.data_ptr()), c_void_p(primals_190.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(primals_191.data_ptr()), c_void_p(primals_192.data_ptr()), c_void_p(primals_193.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(primals_194.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(primals_196.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(primals_197.data_ptr()), c_void_p(primals_198.data_ptr()), c_void_p(primals_199.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(primals_200.data_ptr()), c_void_p(primals_201.data_ptr()), c_void_p(primals_202.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(primals_203.data_ptr()), c_void_p(primals_204.data_ptr()), c_void_p(primals_205.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(primals_206.data_ptr()), c_void_p(primals_207.data_ptr()), c_void_p(primals_208.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(primals_209.data_ptr()), c_void_p(primals_210.data_ptr()), c_void_p(primals_211.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(primals_212.data_ptr()), c_void_p(primals_213.data_ptr()), c_void_p(primals_214.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(primals_215.data_ptr()), c_void_p(primals_216.data_ptr()), c_void_p(primals_217.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(primals_218.data_ptr()), c_void_p(primals_219.data_ptr()), c_void_p(primals_220.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(primals_221.data_ptr()), c_void_p(primals_222.data_ptr()), c_void_p(primals_223.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(primals_224.data_ptr()), c_void_p(primals_225.data_ptr()), c_void_p(primals_226.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(primals_227.data_ptr()), c_void_p(primals_228.data_ptr()), c_void_p(primals_229.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(primals_230.data_ptr()), c_void_p(primals_231.data_ptr()), c_void_p(primals_232.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(primals_233.data_ptr()), c_void_p(primals_234.data_ptr()), c_void_p(primals_235.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(primals_236.data_ptr()), c_void_p(primals_237.data_ptr()), c_void_p(primals_238.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(primals_239.data_ptr()), c_void_p(primals_240.data_ptr()), c_void_p(primals_241.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(primals_242.data_ptr()), c_void_p(primals_243.data_ptr()), c_void_p(primals_244.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(primals_245.data_ptr()), c_void_p(primals_246.data_ptr()), c_void_p(primals_247.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(primals_248.data_ptr()), c_void_p(primals_249.data_ptr()), c_void_p(primals_250.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(primals_251.data_ptr()), c_void_p(primals_252.data_ptr()), c_void_p(primals_253.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(primals_254.data_ptr()), c_void_p(primals_255.data_ptr()), c_void_p(primals_256.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(primals_257.data_ptr()), c_void_p(primals_258.data_ptr()), c_void_p(primals_259.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(primals_260.data_ptr()), c_void_p(primals_261.data_ptr()), c_void_p(primals_262.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(primals_263.data_ptr()), c_void_p(primals_264.data_ptr()), c_void_p(primals_265.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(primals_266.data_ptr()), c_void_p(primals_267.data_ptr()), c_void_p(primals_268.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(primals_269.data_ptr()), c_void_p(primals_270.data_ptr()), c_void_p(primals_271.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(primals_272.data_ptr()), c_void_p(primals_273.data_ptr()), c_void_p(primals_274.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(primals_275.data_ptr()), c_void_p(primals_276.data_ptr()), c_void_p(primals_277.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(primals_278.data_ptr()), c_void_p(primals_279.data_ptr()), c_void_p(primals_280.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(primals_281.data_ptr()), c_void_p(primals_282.data_ptr()), c_void_p(primals_283.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(primals_284.data_ptr()), c_void_p(primals_285.data_ptr()), c_void_p(primals_286.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(primals_287.data_ptr()), c_void_p(primals_288.data_ptr()), c_void_p(primals_289.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(primals_290.data_ptr()), c_void_p(primals_291.data_ptr()), c_void_p(primals_292.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(primals_293.data_ptr()), c_void_p(primals_294.data_ptr()), c_void_p(primals_295.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(primals_296.data_ptr()), c_void_p(primals_297.data_ptr()), c_void_p(primals_298.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(primals_299.data_ptr()), c_void_p(primals_300.data_ptr()), c_void_p(primals_301.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(primals_302.data_ptr()), c_void_p(primals_303.data_ptr()), c_void_p(primals_304.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(primals_305.data_ptr()), c_void_p(primals_306.data_ptr()), c_void_p(primals_307.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(primals_308.data_ptr()), c_void_p(primals_309.data_ptr()), c_void_p(primals_310.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(primals_311.data_ptr()), c_void_p(primals_312.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(primals_175.data_ptr()), c_void_p(primals_176.data_ptr()), c_void_p(primals_177.data_ptr()), c_void_p(primals_178.data_ptr()), c_void_p(primals_179.data_ptr()), c_void_p(primals_180.data_ptr()), c_void_p(primals_181.data_ptr()), c_void_p(primals_182.data_ptr()), c_void_p(primals_183.data_ptr()), c_void_p(primals_184.data_ptr()), c_void_p(primals_185.data_ptr()), c_void_p(primals_186.data_ptr()), c_void_p(primals_187.data_ptr()), c_void_p(primals_188.data_ptr()), c_void_p(primals_189.data_ptr()), c_void_p(primals_190.data_ptr()), c_void_p(primals_191.data_ptr()), c_void_p(primals_192.data_ptr()), c_void_p(primals_193.data_ptr()), c_void_p(primals_194.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(primals_196.data_ptr()), c_void_p(primals_197.data_ptr()), c_void_p(primals_198.data_ptr()), c_void_p(primals_199.data_ptr()), c_void_p(primals_200.data_ptr()), c_void_p(primals_201.data_ptr()), c_void_p(primals_202.data_ptr()), c_void_p(primals_203.data_ptr()), c_void_p(primals_204.data_ptr()), c_void_p(primals_205.data_ptr()), c_void_p(primals_206.data_ptr()), c_void_p(primals_207.data_ptr()), c_void_p(primals_208.data_ptr()), c_void_p(primals_209.data_ptr()), c_void_p(primals_210.data_ptr()), c_void_p(primals_211.data_ptr()), c_void_p(primals_212.data_ptr()), c_void_p(primals_213.data_ptr()), c_void_p(primals_214.data_ptr()), c_void_p(primals_215.data_ptr()), c_void_p(primals_216.data_ptr()), c_void_p(primals_217.data_ptr()), c_void_p(primals_218.data_ptr()), c_void_p(primals_219.data_ptr()), c_void_p(primals_220.data_ptr()), c_void_p(primals_221.data_ptr()), c_void_p(primals_222.data_ptr()), c_void_p(primals_223.data_ptr()), c_void_p(primals_224.data_ptr()), c_void_p(primals_225.data_ptr()), c_void_p(primals_226.data_ptr()), c_void_p(primals_227.data_ptr()), c_void_p(primals_228.data_ptr()), c_void_p(primals_229.data_ptr()), c_void_p(primals_230.data_ptr()), c_void_p(primals_231.data_ptr()), c_void_p(primals_232.data_ptr()), c_void_p(primals_233.data_ptr()), c_void_p(primals_234.data_ptr()), c_void_p(primals_235.data_ptr()), c_void_p(primals_236.data_ptr()), c_void_p(primals_237.data_ptr()), c_void_p(primals_238.data_ptr()), c_void_p(primals_239.data_ptr()), c_void_p(primals_240.data_ptr()), c_void_p(primals_241.data_ptr()), c_void_p(primals_242.data_ptr()), c_void_p(primals_243.data_ptr()), c_void_p(primals_244.data_ptr()), c_void_p(primals_245.data_ptr()), c_void_p(primals_246.data_ptr()), c_void_p(primals_247.data_ptr()), c_void_p(primals_248.data_ptr()), c_void_p(primals_249.data_ptr()), c_void_p(primals_250.data_ptr()), c_void_p(primals_251.data_ptr()), c_void_p(primals_252.data_ptr()), c_void_p(primals_253.data_ptr()), c_void_p(primals_254.data_ptr()), c_void_p(primals_255.data_ptr()), c_void_p(primals_256.data_ptr()), c_void_p(primals_257.data_ptr()), c_void_p(primals_258.data_ptr()), c_void_p(primals_259.data_ptr()), c_void_p(primals_260.data_ptr()), c_void_p(primals_261.data_ptr()), c_void_p(primals_262.data_ptr()), c_void_p(primals_263.data_ptr()), c_void_p(primals_264.data_ptr()), c_void_p(primals_265.data_ptr()), c_void_p(primals_266.data_ptr()), c_void_p(primals_267.data_ptr()), c_void_p(primals_268.data_ptr()), c_void_p(primals_269.data_ptr()), c_void_p(primals_270.data_ptr()), c_void_p(primals_271.data_ptr()), c_void_p(primals_272.data_ptr()), c_void_p(primals_273.data_ptr()), c_void_p(primals_274.data_ptr()), c_void_p(primals_275.data_ptr()), c_void_p(primals_276.data_ptr()), c_void_p(primals_277.data_ptr()), c_void_p(primals_278.data_ptr()), c_void_p(primals_279.data_ptr()), c_void_p(primals_280.data_ptr()), c_void_p(primals_281.data_ptr()), c_void_p(primals_282.data_ptr()), c_void_p(primals_283.data_ptr()), c_void_p(primals_284.data_ptr()), c_void_p(primals_285.data_ptr()), c_void_p(primals_286.data_ptr()), c_void_p(primals_287.data_ptr()), c_void_p(primals_288.data_ptr()), c_void_p(primals_289.data_ptr()), c_void_p(primals_290.data_ptr()), c_void_p(primals_291.data_ptr()), c_void_p(primals_292.data_ptr()), c_void_p(primals_293.data_ptr()), c_void_p(primals_294.data_ptr()), c_void_p(primals_295.data_ptr()), c_void_p(primals_296.data_ptr()), c_void_p(primals_297.data_ptr()), c_void_p(primals_298.data_ptr()), c_void_p(primals_299.data_ptr()), c_void_p(primals_300.data_ptr()), c_void_p(primals_301.data_ptr()), c_void_p(primals_302.data_ptr()), c_void_p(primals_303.data_ptr()), c_void_p(primals_304.data_ptr()), c_void_p(primals_305.data_ptr()), c_void_p(primals_306.data_ptr()), c_void_p(primals_307.data_ptr()), c_void_p(primals_308.data_ptr()), c_void_p(primals_309.data_ptr()), c_void_p(primals_310.data_ptr()), c_void_p(primals_311.data_ptr()), c_void_p(primals_312.data_ptr()))
    del buf123
    del buf230
    del buf257
    del buf284
    del buf372
    del buf380
    del buf388
    del buf396
    del buf404
    del buf412
    del buf420
    del buf428
    del buf436
    del buf444
    del buf452
    del buf460
    del buf468
    del buf476
    del buf484
    del buf492
    del buf500
    del buf508
    del buf516
    del buf524
    del buf532
    del buf540
    del buf548
    del buf556
    del buf564
    del buf572
    del buf580
    del buf588
    del buf596
    del buf604
    del buf612
    del buf620
    del buf628
    del buf636
    del buf644
    del buf652
    del buf660
    del buf668
    del buf676
    del buf684
    del buf692
    del buf700
    del buf708
    del buf716
    del buf724
    del buf73
    del buf732
    del buf98
    del primals_175
    del primals_176
    del primals_177
    del primals_178
    del primals_179
    del primals_180
    del primals_181
    del primals_182
    del primals_183
    del primals_184
    del primals_185
    del primals_186
    del primals_187
    del primals_188
    del primals_189
    del primals_190
    del primals_191
    del primals_192
    del primals_193
    del primals_194
    del primals_195
    del primals_196
    del primals_197
    del primals_198
    del primals_199
    del primals_200
    del primals_201
    del primals_202
    del primals_203
    del primals_204
    del primals_205
    del primals_206
    del primals_207
    del primals_208
    del primals_209
    del primals_210
    del primals_211
    del primals_212
    del primals_213
    del primals_214
    del primals_215
    del primals_216
    del primals_217
    del primals_218
    del primals_219
    del primals_220
    del primals_221
    del primals_222
    del primals_223
    del primals_224
    del primals_225
    del primals_226
    del primals_227
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
    return (buf357, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, buf0, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_108, primals_110, primals_111, primals_112, primals_113, primals_115, primals_117, primals_118, primals_119, primals_120, primals_122, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_141, primals_143, primals_144, primals_145, primals_146, primals_148, primals_150, primals_151, primals_152, primals_153, primals_155, primals_157, primals_158, primals_159, primals_160, primals_162, primals_164, primals_165, primals_166, primals_167, primals_169, primals_171, primals_172, primals_173, buf1, buf2, buf6, buf7, buf8, buf9, buf13, buf14, buf15, buf19, buf20, buf21, buf25, buf26, buf27, buf31, buf32, buf33, buf37, buf38, buf39, buf43, buf44, buf45, buf49, buf50, buf51, buf55, buf56, buf57, buf61, buf62, buf63, buf67, buf68, buf70, buf72, buf74, buf75, buf76, buf80, buf81, buf82, buf86, buf87, buf88, buf92, buf93, buf95, buf97, buf99, buf100, buf101, buf105, buf106, buf107, buf111, buf112, buf113, buf117, buf118, buf120, buf122, buf124, buf125, buf126, buf130, buf131, buf132, buf136, buf137, buf138, buf139, buf143, buf144, buf145, buf146, buf150, buf151, buf152, buf156, buf157, buf158, buf159, buf163, buf164, buf165, buf166, buf170, buf171, buf172, buf176, buf177, buf178, buf179, buf183, buf184, buf185, buf186, buf190, buf191, buf192, buf196, buf197, buf198, buf199, buf203, buf204, buf205, buf206, buf210, buf211, buf212, buf216, buf217, buf218, buf219, buf223, buf224, buf225, buf227, buf229, buf231, buf232, buf233, buf237, buf238, buf239, buf243, buf244, buf245, buf246, buf250, buf251, buf252, buf254, buf256, buf258, buf259, buf260, buf264, buf265, buf266, buf270, buf271, buf272, buf273, buf277, buf278, buf279, buf281, buf283, buf285, buf286, buf287, buf291, buf292, buf293, buf297, buf298, buf299, buf300, buf304, buf305, buf306, buf308, buf310, buf312, buf313, buf314, buf318, buf319, buf320, buf324, buf325, buf326, buf327, buf331, buf332, buf333, buf335, buf337, buf339, buf340, buf341, buf345, buf346, buf347, buf351, buf352, buf354, buf355, buf356, reinterpret_tensor(primals_93, (1000, 1280), (1280, 1), 0), reinterpret_tensor(buf348, (1, 960, 1, 1), (960, 1, 1, 1), 0), reinterpret_tensor(buf342, (1, 160, 1, 1), (160, 1, 1, 1), 0), buf358, reinterpret_tensor(buf328, (1, 960, 1, 1), (960, 1, 1, 1), 0), reinterpret_tensor(buf321, (1, 960, 1, 1), (960, 1, 1, 1), 0), reinterpret_tensor(buf315, (1, 160, 1, 1), (160, 1, 1, 1), 0), buf359, reinterpret_tensor(buf301, (1, 960, 1, 1), (960, 1, 1, 1), 0), reinterpret_tensor(buf294, (1, 960, 1, 1), (960, 1, 1, 1), 0), reinterpret_tensor(buf288, (1, 160, 1, 1), (160, 1, 1, 1), 0), buf360, reinterpret_tensor(buf274, (1, 672, 1, 1), (672, 1, 1, 1), 0), reinterpret_tensor(buf267, (1, 672, 1, 1), (672, 1, 1, 1), 0), reinterpret_tensor(buf261, (1, 112, 1, 1), (112, 1, 1, 1), 0), buf361, reinterpret_tensor(buf247, (1, 672, 1, 1), (672, 1, 1, 1), 0), reinterpret_tensor(buf240, (1, 672, 1, 1), (672, 1, 1, 1), 0), reinterpret_tensor(buf234, (1, 112, 1, 1), (112, 1, 1, 1), 0), buf362, reinterpret_tensor(buf220, (1, 480, 1, 1), (480, 1, 1, 1), 0), reinterpret_tensor(buf213, (1, 480, 1, 1), (480, 1, 1, 1), 0), reinterpret_tensor(buf207, (1, 80, 1, 1), (80, 1, 1, 1), 0), reinterpret_tensor(buf200, (1, 184, 1, 1), (184, 1, 1, 1), 0), reinterpret_tensor(buf193, (1, 184, 1, 1), (184, 1, 1, 1), 0), reinterpret_tensor(buf187, (1, 80, 1, 1), (80, 1, 1, 1), 0), reinterpret_tensor(buf180, (1, 184, 1, 1), (184, 1, 1, 1), 0), reinterpret_tensor(buf173, (1, 184, 1, 1), (184, 1, 1, 1), 0), reinterpret_tensor(buf167, (1, 80, 1, 1), (80, 1, 1, 1), 0), reinterpret_tensor(buf160, (1, 200, 1, 1), (200, 1, 1, 1), 0), reinterpret_tensor(buf153, (1, 200, 1, 1), (200, 1, 1, 1), 0), reinterpret_tensor(buf147, (1, 80, 1, 1), (80, 1, 1, 1), 0), reinterpret_tensor(buf140, (1, 240, 1, 1), (240, 1, 1, 1), 0), reinterpret_tensor(buf133, (1, 240, 1, 1), (240, 1, 1, 1), 0), reinterpret_tensor(buf127, (1, 40, 1, 1), (40, 1, 1, 1), 0), buf363, reinterpret_tensor(buf114, (1, 120, 1, 1), (120, 1, 1, 1), 0), reinterpret_tensor(buf108, (1, 120, 1, 1), (120, 1, 1, 1), 0), reinterpret_tensor(buf102, (1, 40, 1, 1), (40, 1, 1, 1), 0), buf364, reinterpret_tensor(buf89, (1, 120, 1, 1), (120, 1, 1, 1), 0), reinterpret_tensor(buf83, (1, 120, 1, 1), (120, 1, 1, 1), 0), reinterpret_tensor(buf77, (1, 40, 1, 1), (40, 1, 1, 1), 0), buf365, reinterpret_tensor(buf64, (1, 72, 1, 1), (72, 1, 1, 1), 0), reinterpret_tensor(buf58, (1, 72, 1, 1), (72, 1, 1, 1), 0), reinterpret_tensor(buf52, (1, 24, 1, 1), (24, 1, 1, 1), 0), reinterpret_tensor(buf46, (1, 72, 1, 1), (72, 1, 1, 1), 0), reinterpret_tensor(buf40, (1, 72, 1, 1), (72, 1, 1, 1), 0), reinterpret_tensor(buf34, (1, 24, 1, 1), (24, 1, 1, 1), 0), reinterpret_tensor(buf28, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf22, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf16, (1, 16, 1, 1), (16, 1, 1, 1), 0), reinterpret_tensor(buf10, (1, 16, 1, 1), (16, 1, 1, 1), 0), reinterpret_tensor(buf3, (1, 16, 1, 1), (16, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((1000, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((24, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((72, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((24, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((72, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((24, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((40, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((32, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((120, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((32, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((120, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((200, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((200, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((80, 200, 1, 1), (200, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((184, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((184, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((80, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((184, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((184, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((80, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((120, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((480, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((112, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((672, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((168, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((168, ), (1, ), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((672, 168, 1, 1), (168, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((168, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((168, ), (1, ), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((672, 168, 1, 1), (168, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((160, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((960, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((240, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((960, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((960, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((240, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((960, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((1280, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_176 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_179 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_182 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_185 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_188 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_191 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_194 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_197 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_200 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_203 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_204 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_206 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_207 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_209 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_210 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_212 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_213 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_214 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_215 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_216 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_218 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_219 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_220 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_221 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_222 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_223 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_224 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_225 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_226 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_227 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_228 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_229 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_230 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_231 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_232 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_233 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_234 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_235 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_236 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_237 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_238 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_239 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    primals_240 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    primals_241 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_242 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    primals_243 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    primals_244 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_245 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_246 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_247 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_248 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    primals_249 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    primals_250 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_251 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    primals_252 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    primals_253 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_254 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_255 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_256 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_257 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    primals_258 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    primals_259 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_260 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    primals_261 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    primals_262 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_263 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_264 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_265 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_266 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_267 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_268 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_269 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_270 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_271 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_272 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_273 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_274 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_275 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_276 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_277 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_278 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_279 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_280 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_281 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_282 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_283 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_284 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_285 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_286 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_287 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_288 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_289 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_290 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_291 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_292 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_293 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_294 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_295 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_296 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_297 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_298 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_299 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_300 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_301 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_302 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_303 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_304 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_305 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_306 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_307 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_308 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_309 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_310 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_311 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_312 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_313 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mobilenetv3_large_100', benchmark_compiled_module)
