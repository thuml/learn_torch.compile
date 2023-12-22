
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


cpp_fused__native_batch_norm_legit_functional_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
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
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8,
                       float* out_ptr9,
                       float* out_ptr10,
                       float* out_ptr11,
                       float* out_ptr12,
                       float* out_ptr13,
                       float* out_ptr14,
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
                       float* out_ptr25)
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (25L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr2 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr3 + static_cast<long>(x0));
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
                    auto tmp0 = in_ptr1[static_cast<long>(x0 + (25L*x1))];
                    tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                }
                out_ptr2[static_cast<long>(x0)] = tmp_acc0.mean;
                out_ptr3[static_cast<long>(x0)] = tmp_acc0.m2;
            }
        }
    }
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (30L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr4 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr5 + static_cast<long>(x0));
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
                    auto tmp0 = in_ptr2[static_cast<long>(x0 + (30L*x1))];
                    tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                }
                out_ptr4[static_cast<long>(x0)] = tmp_acc0.mean;
                out_ptr5[static_cast<long>(x0)] = tmp_acc0.m2;
            }
        }
    }
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (36L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr6 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr7 + static_cast<long>(x0));
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
                    auto tmp0 = in_ptr3[static_cast<long>(x0 + (36L*x1))];
                    tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                }
                out_ptr6[static_cast<long>(x0)] = tmp_acc0.mean;
                out_ptr7[static_cast<long>(x0)] = tmp_acc0.m2;
            }
        }
    }
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (42L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr8 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr9 + static_cast<long>(x0));
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
                    auto tmp0 = in_ptr4[static_cast<long>(x0 + (42L*x1))];
                    tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                }
                out_ptr8[static_cast<long>(x0)] = tmp_acc0.mean;
                out_ptr9[static_cast<long>(x0)] = tmp_acc0.m2;
            }
        }
    }
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (47L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr10 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr11 + static_cast<long>(x0));
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
                    auto tmp0 = in_ptr5[static_cast<long>(x0 + (47L*x1))];
                    tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                }
                out_ptr10[static_cast<long>(x0)] = tmp_acc0.mean;
                out_ptr11[static_cast<long>(x0)] = tmp_acc0.m2;
            }
        }
    }
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (53L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr12 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr13 + static_cast<long>(x0));
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
                    auto tmp0 = in_ptr6[static_cast<long>(x0 + (53L*x1))];
                    tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                }
                out_ptr12[static_cast<long>(x0)] = tmp_acc0.mean;
                out_ptr13[static_cast<long>(x0)] = tmp_acc0.m2;
            }
        }
    }
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (58L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr14 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr15 + static_cast<long>(x0));
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
                    auto tmp0 = in_ptr7[static_cast<long>(x0 + (58L*x1))];
                    tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                }
                out_ptr14[static_cast<long>(x0)] = tmp_acc0.mean;
                out_ptr15[static_cast<long>(x0)] = tmp_acc0.m2;
            }
        }
    }
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0 + (64L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr16 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr17 + static_cast<long>(x0));
            }
        }
    }
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0 + (70L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr18 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr19 + static_cast<long>(x0));
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
                    auto tmp0 = in_ptr9[static_cast<long>(x0 + (70L*x1))];
                    tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                }
                out_ptr18[static_cast<long>(x0)] = tmp_acc0.mean;
                out_ptr19[static_cast<long>(x0)] = tmp_acc0.m2;
            }
        }
    }
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0 + (75L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr20 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr21 + static_cast<long>(x0));
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
                    auto tmp0 = in_ptr10[static_cast<long>(x0 + (75L*x1))];
                    tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                }
                out_ptr20[static_cast<long>(x0)] = tmp_acc0.mean;
                out_ptr21[static_cast<long>(x0)] = tmp_acc0.m2;
            }
        }
    }
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x0 + (81L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr22 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr23 + static_cast<long>(x0));
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
                    auto tmp0 = in_ptr11[static_cast<long>(x0 + (81L*x1))];
                    tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                }
                out_ptr22[static_cast<long>(x0)] = tmp_acc0.mean;
                out_ptr23[static_cast<long>(x0)] = tmp_acc0.m2;
            }
        }
    }
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x0 + (87L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr24 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr25 + static_cast<long>(x0));
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
                    auto tmp0 = in_ptr12[static_cast<long>(x0 + (87L*x1))];
                    tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                }
                out_ptr24[static_cast<long>(x0)] = tmp_acc0.mean;
                out_ptr25[static_cast<long>(x0)] = tmp_acc0.m2;
            }
        }
    }
}
''')


cpp_fused_convolution_backward_div_mul_native_batch_norm_backward_sum_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1000L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1000L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1280L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1280L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1280L*x2) + (62720L*x1)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (1280L*x2) + (62720L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(49.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp8 = tmp6 - tmp7;
                            auto tmp9 = tmp5 * tmp8;
                            tmp_acc0_vec = tmp_acc0_vec + tmp5;
                            tmp_acc1_vec = tmp_acc1_vec + tmp9;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1280L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1280L); x2+=static_cast<long>(8L))
                    {
                        float tmp24[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1280L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1280L*x1) + (1280L*x1_inner) + (62720L*x0)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1280L*x1) + (1280L*x1_inner) + (62720L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x2));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                            auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                            auto tmp1 = static_cast<float>(49.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp5 = tmp3 * tmp4;
                            auto tmp8 = tmp6 - tmp7;
                            auto tmp10 = static_cast<float>(0.002551020408163265);
                            auto tmp11 = at::vec::Vectorized<float>(tmp10);
                            auto tmp12 = tmp9 * tmp11;
                            auto tmp14 = tmp13 * tmp13;
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp8 * tmp15;
                            auto tmp17 = tmp5 - tmp16;
                            auto tmp19 = tmp18 * tmp11;
                            auto tmp20 = tmp17 - tmp19;
                            auto tmp22 = tmp13 * tmp21;
                            auto tmp23 = tmp20 * tmp22;
                            tmp23.store(tmp24 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp24, 8, out_ptr4 + static_cast<long>(x1 + (49L*x2) + (62720L*x0)), static_cast<long>(49L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1280L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (1280L*x0))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (1280L*x1) + (62720L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x2 + (1280L*x1) + (62720L*x0))];
                        auto tmp6 = in_ptr4[static_cast<long>(x2)];
                        auto tmp8 = out_ptr2[static_cast<long>(x2)];
                        auto tmp11 = in_ptr5[static_cast<long>(x2)];
                        auto tmp16 = out_ptr1[static_cast<long>(x2)];
                        auto tmp19 = in_ptr6[static_cast<long>(x2)];
                        auto tmp1 = static_cast<float>(49.0);
                        auto tmp2 = tmp0 / tmp1;
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                        auto tmp9 = static_cast<float>(0.002551020408163265);
                        auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                        auto tmp12 = decltype(tmp11)(tmp11 * tmp11);
                        auto tmp13 = decltype(tmp10)(tmp10 * tmp12);
                        auto tmp14 = decltype(tmp7)(tmp7 * tmp13);
                        auto tmp15 = decltype(tmp4)(tmp4 - tmp14);
                        auto tmp17 = decltype(tmp16)(tmp16 * tmp9);
                        auto tmp18 = decltype(tmp15)(tmp15 - tmp17);
                        auto tmp20 = decltype(tmp11)(tmp11 * tmp19);
                        auto tmp21 = decltype(tmp18)(tmp18 * tmp20);
                        out_ptr4[static_cast<long>(x1 + (49L*x2) + (62720L*x0))] = tmp21;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_2 = async_compile.cpp('''
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(185L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp23 = in_ptr1[static_cast<long>(x0 + (185L*x1))];
                    auto tmp24 = in_ptr2[static_cast<long>(x0)];
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = static_cast<long>(174);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = in_ptr0[static_cast<long>(x0 + (185L*x1))];
                        return tmp4;
                    }
                    ;
                    auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                    auto tmp6 = static_cast<float>(0.0);
                    auto tmp7 = tmp2 ? tmp5 : tmp6;
                    auto tmp8 = tmp0 < tmp1;
                    auto tmp9 = [&]
                    {
                        auto tmp10 = in_ptr0[static_cast<long>(x0 + (185L*x1))];
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp8 ? tmp9() : static_cast<decltype(tmp9())>(0.0);
                    auto tmp12 = tmp8 ? tmp11 : tmp6;
                    auto tmp13 = decltype(tmp7)(tmp7 + tmp12);
                    auto tmp14 = [&]
                    {
                        auto tmp15 = in_ptr0[static_cast<long>(x0 + (185L*x1))];
                        return tmp15;
                    }
                    ;
                    auto tmp16 = tmp2 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                    auto tmp17 = tmp2 ? tmp16 : tmp6;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr0[static_cast<long>(x0 + (185L*x1))];
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp8 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp21 = tmp8 ? tmp20 : tmp6;
                    auto tmp22 = decltype(tmp17)(tmp17 + tmp21);
                    auto tmp25 = decltype(tmp23)(tmp23 - tmp24);
                    auto tmp26 = decltype(tmp22)(tmp22 * tmp25);
                    tmp_acc0 = tmp_acc0 + tmp13;
                    tmp_acc1 = tmp_acc1 + tmp26;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(184L); x0<static_cast<long>(185L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr3[static_cast<long>(x0)];
            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
            out_ptr2[static_cast<long>(x0)] = tmp2;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(185L); x2+=static_cast<long>(1L))
                {
                    auto tmp14 = in_ptr1[static_cast<long>(x2 + (185L*x1) + (9065L*x0))];
                    auto tmp15 = in_ptr2[static_cast<long>(x2)];
                    auto tmp17 = out_ptr1[static_cast<long>(x2)];
                    auto tmp20 = in_ptr3[static_cast<long>(x2)];
                    auto tmp25 = out_ptr0[static_cast<long>(x2)];
                    auto tmp28 = in_ptr4[static_cast<long>(x2)];
                    auto tmp0 = c10::convert<long>(x2);
                    auto tmp1 = static_cast<long>(174);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = in_ptr0[static_cast<long>(x2 + (185L*x1) + (9065L*x0))];
                        return tmp4;
                    }
                    ;
                    auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                    auto tmp6 = static_cast<float>(0.0);
                    auto tmp7 = tmp2 ? tmp5 : tmp6;
                    auto tmp8 = tmp0 < tmp1;
                    auto tmp9 = [&]
                    {
                        auto tmp10 = in_ptr0[static_cast<long>(x2 + (185L*x1) + (9065L*x0))];
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp8 ? tmp9() : static_cast<decltype(tmp9())>(0.0);
                    auto tmp12 = tmp8 ? tmp11 : tmp6;
                    auto tmp13 = decltype(tmp7)(tmp7 + tmp12);
                    auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                    auto tmp18 = static_cast<float>(0.002551020408163265);
                    auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                    auto tmp21 = decltype(tmp20)(tmp20 * tmp20);
                    auto tmp22 = decltype(tmp19)(tmp19 * tmp21);
                    auto tmp23 = decltype(tmp16)(tmp16 * tmp22);
                    auto tmp24 = decltype(tmp13)(tmp13 - tmp23);
                    auto tmp26 = decltype(tmp25)(tmp25 * tmp18);
                    auto tmp27 = decltype(tmp24)(tmp24 - tmp26);
                    auto tmp29 = decltype(tmp20)(tmp20 * tmp28);
                    auto tmp30 = decltype(tmp27)(tmp27 * tmp29);
                    out_ptr3[static_cast<long>(x1 + (49L*x2) + (9065L*x0))] = tmp30;
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1044L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (1044L*x2) + (51156L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (1044L*x0))];
                            auto tmp9 = in_ptr2[static_cast<long>(x1 + (1044L*x2) + (51156L*x0))];
                            auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = tmp3 <= tmp4;
                            auto tmp6 = static_cast<float>(6.0);
                            auto tmp7 = tmp3 >= tmp6;
                            auto tmp8 = decltype(tmp5)(tmp5 | tmp7);
                            auto tmp10 = tmp8 ? tmp4 : tmp9;
                            auto tmp11 = decltype(tmp10)(tmp10 * tmp0);
                            tmp_acc0 = tmp_acc0 + tmp11;
                        }
                        out_ptr0[static_cast<long>(x1 + (1044L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8352L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_threshold_backward_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (87L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (87L*x1)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (87L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp9 = tmp5 * tmp8;
                    tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    tmp_acc1_vec = tmp_acc1_vec + tmp9;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(80L); x0<static_cast<long>(87L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (87L*x1))];
                    auto tmp3 = in_ptr1[static_cast<long>(x0 + (87L*x1))];
                    auto tmp5 = in_ptr2[static_cast<long>(x0 + (87L*x1))];
                    auto tmp6 = in_ptr3[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    auto tmp4 = tmp2 ? tmp1 : tmp3;
                    auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                    auto tmp8 = decltype(tmp4)(tmp4 * tmp7);
                    tmp_acc0 = tmp_acc0 + tmp4;
                    tmp_acc1 = tmp_acc1 + tmp8;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(8.0);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 / tmp3;
            auto tmp5 = static_cast<float>(1e-05);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 + tmp6;
            auto tmp8 = tmp7.rsqrt();
            auto tmp9 = tmp0 * tmp8;
            tmp9.store(out_ptr2 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(80L); x0<static_cast<long>(87L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr4[static_cast<long>(x0)];
            auto tmp2 = static_cast<float>(8.0);
            auto tmp3 = tmp1 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
            auto tmp6 = 1 / std::sqrt(tmp5);
            auto tmp7 = decltype(tmp0)(tmp0 * tmp6);
            out_ptr2[static_cast<long>(x0)] = tmp7;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (87L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (87L*x0)));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (87L*x0)));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp25 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                auto tmp8 = tmp6 - tmp7;
                auto tmp10 = static_cast<float>(0.125);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp14 = static_cast<float>(8.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 / tmp15;
                auto tmp17 = static_cast<float>(1e-05);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 + tmp18;
                auto tmp20 = tmp19.rsqrt();
                auto tmp21 = tmp20 * tmp20;
                auto tmp22 = tmp12 * tmp21;
                auto tmp23 = tmp8 * tmp22;
                auto tmp24 = tmp5 - tmp23;
                auto tmp26 = tmp25 * tmp11;
                auto tmp27 = tmp24 - tmp26;
                auto tmp29 = tmp20 * tmp28;
                auto tmp30 = tmp27 * tmp29;
                tmp30.store(in_out_ptr0 + static_cast<long>(x1 + (87L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(80L); x1<static_cast<long>(87L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (87L*x0))];
                auto tmp3 = in_out_ptr0[static_cast<long>(x1 + (87L*x0))];
                auto tmp5 = in_ptr2[static_cast<long>(x1 + (87L*x0))];
                auto tmp6 = in_ptr3[static_cast<long>(x1)];
                auto tmp8 = out_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr4[static_cast<long>(x1)];
                auto tmp21 = out_ptr0[static_cast<long>(x1)];
                auto tmp24 = in_ptr5[static_cast<long>(x1)];
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp4 = tmp2 ? tmp1 : tmp3;
                auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                auto tmp9 = static_cast<float>(0.125);
                auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                auto tmp12 = static_cast<float>(8.0);
                auto tmp13 = tmp11 / tmp12;
                auto tmp14 = static_cast<float>(1e-05);
                auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                auto tmp16 = 1 / std::sqrt(tmp15);
                auto tmp17 = decltype(tmp16)(tmp16 * tmp16);
                auto tmp18 = decltype(tmp10)(tmp10 * tmp17);
                auto tmp19 = decltype(tmp7)(tmp7 * tmp18);
                auto tmp20 = decltype(tmp4)(tmp4 - tmp19);
                auto tmp22 = decltype(tmp21)(tmp21 * tmp9);
                auto tmp23 = decltype(tmp20)(tmp20 - tmp22);
                auto tmp25 = decltype(tmp16)(tmp16 * tmp24);
                auto tmp26 = decltype(tmp23)(tmp23 * tmp25);
                in_out_ptr0[static_cast<long>(x1 + (87L*x0))] = tmp26;
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0)
{
    auto in_ptr2 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1044L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x0 + (1044L*x2) + (51156L*x1))];
                            auto tmp1 = in_ptr1[static_cast<long>(x0 + (1044L*x1))];
                            auto tmp9 = in_ptr2[static_cast<long>(x0 + (1044L*x2) + (51156L*x1))];
                            auto tmp12 = in_ptr3[static_cast<long>(x0 + (1044L*x1))];
                            auto tmp16 = in_ptr4[static_cast<long>(x0 + (1044L*x2) + (51156L*x1))];
                            auto tmp17 = in_ptr5[static_cast<long>(x0)];
                            auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = tmp3 <= tmp4;
                            auto tmp6 = static_cast<float>(6.0);
                            auto tmp7 = tmp3 >= tmp6;
                            auto tmp8 = decltype(tmp5)(tmp5 | tmp7);
                            auto tmp10 = tmp8 ? tmp4 : tmp9;
                            auto tmp11 = decltype(tmp10)(tmp10 * tmp2);
                            auto tmp13 = static_cast<float>(49.0);
                            auto tmp14 = tmp12 / tmp13;
                            auto tmp15 = decltype(tmp11)(tmp11 + tmp14);
                            auto tmp18 = decltype(tmp16)(tmp16 - tmp17);
                            auto tmp19 = decltype(tmp15)(tmp15 * tmp18);
                            tmp_acc0 = tmp_acc0 + tmp15;
                            tmp_acc1 = tmp_acc1 + tmp19;
                        }
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1044L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (1044L*x1) + (51156L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (1044L*x0))];
                        auto tmp9 = in_out_ptr0[static_cast<long>(x2 + (1044L*x1) + (51156L*x0))];
                        auto tmp12 = in_ptr3[static_cast<long>(x2 + (1044L*x0))];
                        auto tmp16 = in_ptr4[static_cast<long>(x2 + (1044L*x1) + (51156L*x0))];
                        auto tmp17 = in_ptr5[static_cast<long>(x2)];
                        auto tmp19 = out_ptr1[static_cast<long>(x2)];
                        auto tmp22 = in_ptr6[static_cast<long>(x2)];
                        auto tmp27 = out_ptr0[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                        auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = tmp3 <= tmp4;
                        auto tmp6 = static_cast<float>(6.0);
                        auto tmp7 = tmp3 >= tmp6;
                        auto tmp8 = decltype(tmp5)(tmp5 | tmp7);
                        auto tmp10 = tmp8 ? tmp4 : tmp9;
                        auto tmp11 = decltype(tmp10)(tmp10 * tmp2);
                        auto tmp13 = static_cast<float>(49.0);
                        auto tmp14 = tmp12 / tmp13;
                        auto tmp15 = decltype(tmp11)(tmp11 + tmp14);
                        auto tmp18 = decltype(tmp16)(tmp16 - tmp17);
                        auto tmp20 = static_cast<float>(0.002551020408163265);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp23 = decltype(tmp22)(tmp22 * tmp22);
                        auto tmp24 = decltype(tmp21)(tmp21 * tmp23);
                        auto tmp25 = decltype(tmp18)(tmp18 * tmp24);
                        auto tmp26 = decltype(tmp15)(tmp15 - tmp25);
                        auto tmp28 = decltype(tmp27)(tmp27 * tmp20);
                        auto tmp29 = decltype(tmp26)(tmp26 - tmp28);
                        in_out_ptr0[static_cast<long>(x2 + (1044L*x1) + (51156L*x0))] = tmp29;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1040L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(1040L); x0<static_cast<long>(1044L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    in_out_ptr1[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1040L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1044L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (1044L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(1040L); x1<static_cast<long>(1044L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (1044L*x0))];
                    auto tmp1 = in_ptr6[static_cast<long>(x1)];
                    auto tmp2 = in_ptr7[static_cast<long>(x1)];
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                    in_out_ptr0[static_cast<long>(x1 + (1044L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1040L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1044L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1044L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1044L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(1040L); x0<static_cast<long>(1044L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (1044L*x1))];
                        auto tmp1 = in_ptr1[static_cast<long>(x0 + (1044L*x1))];
                        auto tmp3 = in_ptr2[static_cast<long>(x0 + (1044L*x1))];
                        auto tmp4 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = decltype(tmp2)(tmp2 * tmp5);
                        tmp_acc0 = tmp_acc0 + tmp2;
                        tmp_acc1 = tmp_acc1 + tmp6;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1040L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(1040L); x0<static_cast<long>(1044L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    out_ptr2[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1040L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1044L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1044L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1044L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.002551020408163265);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (1044L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(1040L); x1<static_cast<long>(1044L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (1044L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (1044L*x0))];
                    auto tmp3 = in_ptr2[static_cast<long>(x1 + (1044L*x0))];
                    auto tmp4 = in_ptr3[static_cast<long>(x1)];
                    auto tmp6 = out_ptr1[static_cast<long>(x1)];
                    auto tmp9 = in_ptr4[static_cast<long>(x1)];
                    auto tmp14 = out_ptr0[static_cast<long>(x1)];
                    auto tmp17 = in_ptr5[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                    auto tmp7 = static_cast<float>(0.002551020408163265);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp10 = decltype(tmp9)(tmp9 * tmp9);
                    auto tmp11 = decltype(tmp8)(tmp8 * tmp10);
                    auto tmp12 = decltype(tmp5)(tmp5 * tmp11);
                    auto tmp13 = decltype(tmp2)(tmp2 - tmp12);
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp7);
                    auto tmp16 = decltype(tmp13)(tmp13 - tmp15);
                    auto tmp18 = decltype(tmp9)(tmp9 * tmp17);
                    auto tmp19 = decltype(tmp16)(tmp16 * tmp18);
                    in_out_ptr0[static_cast<long>(x1 + (1044L*x0))] = tmp19;
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_7 = async_compile.cpp('''
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(174L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp31 = in_ptr2[static_cast<long>(x0 + (174L*x1))];
                    auto tmp32 = in_ptr3[static_cast<long>(x0)];
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = static_cast<long>(162);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = in_ptr0[static_cast<long>(x0 + (185L*x1))];
                        auto tmp5 = in_ptr1[static_cast<long>(x0 + (174L*x1))];
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                    auto tmp8 = static_cast<float>(0.0);
                    auto tmp9 = tmp2 ? tmp7 : tmp8;
                    auto tmp10 = tmp0 < tmp1;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr0[static_cast<long>(x0 + (185L*x1))];
                        auto tmp13 = in_ptr1[static_cast<long>(x0 + (174L*x1))];
                        auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                        return tmp14;
                    }
                    ;
                    auto tmp15 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp16 = tmp10 ? tmp15 : tmp8;
                    auto tmp17 = decltype(tmp9)(tmp9 + tmp16);
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr0[static_cast<long>(x0 + (185L*x1))];
                        auto tmp20 = in_ptr1[static_cast<long>(x0 + (174L*x1))];
                        auto tmp21 = decltype(tmp19)(tmp19 + tmp20);
                        return tmp21;
                    }
                    ;
                    auto tmp22 = tmp2 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp23 = tmp2 ? tmp22 : tmp8;
                    auto tmp24 = [&]
                    {
                        auto tmp25 = in_ptr0[static_cast<long>(x0 + (185L*x1))];
                        auto tmp26 = in_ptr1[static_cast<long>(x0 + (174L*x1))];
                        auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                        return tmp27;
                    }
                    ;
                    auto tmp28 = tmp10 ? tmp24() : static_cast<decltype(tmp24())>(0.0);
                    auto tmp29 = tmp10 ? tmp28 : tmp8;
                    auto tmp30 = decltype(tmp23)(tmp23 + tmp29);
                    auto tmp33 = decltype(tmp31)(tmp31 - tmp32);
                    auto tmp34 = decltype(tmp30)(tmp30 * tmp33);
                    tmp_acc0 = tmp_acc0 + tmp17;
                    tmp_acc1 = tmp_acc1 + tmp34;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(168L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(168L); x0<static_cast<long>(174L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr4[static_cast<long>(x0)];
            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
            out_ptr2[static_cast<long>(x0)] = tmp2;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(174L); x2+=static_cast<long>(1L))
                {
                    auto tmp18 = in_ptr2[static_cast<long>(x2 + (174L*x1) + (8526L*x0))];
                    auto tmp19 = in_ptr3[static_cast<long>(x2)];
                    auto tmp21 = out_ptr1[static_cast<long>(x2)];
                    auto tmp24 = in_ptr4[static_cast<long>(x2)];
                    auto tmp29 = out_ptr0[static_cast<long>(x2)];
                    auto tmp32 = in_ptr5[static_cast<long>(x2)];
                    auto tmp0 = c10::convert<long>(x2);
                    auto tmp1 = static_cast<long>(162);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = in_ptr0[static_cast<long>(x2 + (185L*x1) + (9065L*x0))];
                        auto tmp5 = in_ptr1[static_cast<long>(x2 + (174L*x1) + (8526L*x0))];
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                    auto tmp8 = static_cast<float>(0.0);
                    auto tmp9 = tmp2 ? tmp7 : tmp8;
                    auto tmp10 = tmp0 < tmp1;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr0[static_cast<long>(x2 + (185L*x1) + (9065L*x0))];
                        auto tmp13 = in_ptr1[static_cast<long>(x2 + (174L*x1) + (8526L*x0))];
                        auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                        return tmp14;
                    }
                    ;
                    auto tmp15 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp16 = tmp10 ? tmp15 : tmp8;
                    auto tmp17 = decltype(tmp9)(tmp9 + tmp16);
                    auto tmp20 = decltype(tmp18)(tmp18 - tmp19);
                    auto tmp22 = static_cast<float>(0.002551020408163265);
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp25 = decltype(tmp24)(tmp24 * tmp24);
                    auto tmp26 = decltype(tmp23)(tmp23 * tmp25);
                    auto tmp27 = decltype(tmp20)(tmp20 * tmp26);
                    auto tmp28 = decltype(tmp17)(tmp17 - tmp27);
                    auto tmp30 = decltype(tmp29)(tmp29 * tmp22);
                    auto tmp31 = decltype(tmp28)(tmp28 - tmp30);
                    auto tmp33 = decltype(tmp24)(tmp24 * tmp32);
                    auto tmp34 = decltype(tmp31)(tmp31 * tmp33);
                    out_ptr3[static_cast<long>(x1 + (49L*x2) + (8526L*x0))] = tmp34;
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(972L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (972L*x2) + (47628L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (972L*x0))];
                            auto tmp9 = in_ptr2[static_cast<long>(x1 + (972L*x2) + (47628L*x0))];
                            auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = tmp3 <= tmp4;
                            auto tmp6 = static_cast<float>(6.0);
                            auto tmp7 = tmp3 >= tmp6;
                            auto tmp8 = decltype(tmp5)(tmp5 | tmp7);
                            auto tmp10 = tmp8 ? tmp4 : tmp9;
                            auto tmp11 = decltype(tmp10)(tmp10 * tmp0);
                            tmp_acc0 = tmp_acc0 + tmp11;
                        }
                        out_ptr0[static_cast<long>(x1 + (972L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(7776L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_threshold_backward_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (81L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (81L*x1)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (81L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp9 = tmp5 * tmp8;
                    tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    tmp_acc1_vec = tmp_acc1_vec + tmp9;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(80L); x0<static_cast<long>(81L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (81L*x1))];
                    auto tmp3 = in_ptr1[static_cast<long>(x0 + (81L*x1))];
                    auto tmp5 = in_ptr2[static_cast<long>(x0 + (81L*x1))];
                    auto tmp6 = in_ptr3[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    auto tmp4 = tmp2 ? tmp1 : tmp3;
                    auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                    auto tmp8 = decltype(tmp4)(tmp4 * tmp7);
                    tmp_acc0 = tmp_acc0 + tmp4;
                    tmp_acc1 = tmp_acc1 + tmp8;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(8.0);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 / tmp3;
            auto tmp5 = static_cast<float>(1e-05);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 + tmp6;
            auto tmp8 = tmp7.rsqrt();
            auto tmp9 = tmp0 * tmp8;
            tmp9.store(out_ptr2 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(80L); x0<static_cast<long>(81L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr4[static_cast<long>(x0)];
            auto tmp2 = static_cast<float>(8.0);
            auto tmp3 = tmp1 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
            auto tmp6 = 1 / std::sqrt(tmp5);
            auto tmp7 = decltype(tmp0)(tmp0 * tmp6);
            out_ptr2[static_cast<long>(x0)] = tmp7;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (81L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (81L*x0)));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (81L*x0)));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp25 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                auto tmp8 = tmp6 - tmp7;
                auto tmp10 = static_cast<float>(0.125);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp14 = static_cast<float>(8.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 / tmp15;
                auto tmp17 = static_cast<float>(1e-05);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 + tmp18;
                auto tmp20 = tmp19.rsqrt();
                auto tmp21 = tmp20 * tmp20;
                auto tmp22 = tmp12 * tmp21;
                auto tmp23 = tmp8 * tmp22;
                auto tmp24 = tmp5 - tmp23;
                auto tmp26 = tmp25 * tmp11;
                auto tmp27 = tmp24 - tmp26;
                auto tmp29 = tmp20 * tmp28;
                auto tmp30 = tmp27 * tmp29;
                tmp30.store(in_out_ptr0 + static_cast<long>(x1 + (81L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(80L); x1<static_cast<long>(81L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (81L*x0))];
                auto tmp3 = in_out_ptr0[static_cast<long>(x1 + (81L*x0))];
                auto tmp5 = in_ptr2[static_cast<long>(x1 + (81L*x0))];
                auto tmp6 = in_ptr3[static_cast<long>(x1)];
                auto tmp8 = out_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr4[static_cast<long>(x1)];
                auto tmp21 = out_ptr0[static_cast<long>(x1)];
                auto tmp24 = in_ptr5[static_cast<long>(x1)];
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp4 = tmp2 ? tmp1 : tmp3;
                auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                auto tmp9 = static_cast<float>(0.125);
                auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                auto tmp12 = static_cast<float>(8.0);
                auto tmp13 = tmp11 / tmp12;
                auto tmp14 = static_cast<float>(1e-05);
                auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                auto tmp16 = 1 / std::sqrt(tmp15);
                auto tmp17 = decltype(tmp16)(tmp16 * tmp16);
                auto tmp18 = decltype(tmp10)(tmp10 * tmp17);
                auto tmp19 = decltype(tmp7)(tmp7 * tmp18);
                auto tmp20 = decltype(tmp4)(tmp4 - tmp19);
                auto tmp22 = decltype(tmp21)(tmp21 * tmp9);
                auto tmp23 = decltype(tmp20)(tmp20 - tmp22);
                auto tmp25 = decltype(tmp16)(tmp16 * tmp24);
                auto tmp26 = decltype(tmp23)(tmp23 * tmp25);
                in_out_ptr0[static_cast<long>(x1 + (81L*x0))] = tmp26;
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0)
{
    auto in_ptr2 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(972L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x0 + (972L*x2) + (47628L*x1))];
                            auto tmp1 = in_ptr1[static_cast<long>(x0 + (972L*x1))];
                            auto tmp9 = in_ptr2[static_cast<long>(x0 + (972L*x2) + (47628L*x1))];
                            auto tmp12 = in_ptr3[static_cast<long>(x0 + (972L*x1))];
                            auto tmp16 = in_ptr4[static_cast<long>(x0 + (972L*x2) + (47628L*x1))];
                            auto tmp17 = in_ptr5[static_cast<long>(x0)];
                            auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = tmp3 <= tmp4;
                            auto tmp6 = static_cast<float>(6.0);
                            auto tmp7 = tmp3 >= tmp6;
                            auto tmp8 = decltype(tmp5)(tmp5 | tmp7);
                            auto tmp10 = tmp8 ? tmp4 : tmp9;
                            auto tmp11 = decltype(tmp10)(tmp10 * tmp2);
                            auto tmp13 = static_cast<float>(49.0);
                            auto tmp14 = tmp12 / tmp13;
                            auto tmp15 = decltype(tmp11)(tmp11 + tmp14);
                            auto tmp18 = decltype(tmp16)(tmp16 - tmp17);
                            auto tmp19 = decltype(tmp15)(tmp15 * tmp18);
                            tmp_acc0 = tmp_acc0 + tmp15;
                            tmp_acc1 = tmp_acc1 + tmp19;
                        }
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(972L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (972L*x1) + (47628L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (972L*x0))];
                        auto tmp9 = in_out_ptr0[static_cast<long>(x2 + (972L*x1) + (47628L*x0))];
                        auto tmp12 = in_ptr3[static_cast<long>(x2 + (972L*x0))];
                        auto tmp16 = in_ptr4[static_cast<long>(x2 + (972L*x1) + (47628L*x0))];
                        auto tmp17 = in_ptr5[static_cast<long>(x2)];
                        auto tmp19 = out_ptr1[static_cast<long>(x2)];
                        auto tmp22 = in_ptr6[static_cast<long>(x2)];
                        auto tmp27 = out_ptr0[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                        auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = tmp3 <= tmp4;
                        auto tmp6 = static_cast<float>(6.0);
                        auto tmp7 = tmp3 >= tmp6;
                        auto tmp8 = decltype(tmp5)(tmp5 | tmp7);
                        auto tmp10 = tmp8 ? tmp4 : tmp9;
                        auto tmp11 = decltype(tmp10)(tmp10 * tmp2);
                        auto tmp13 = static_cast<float>(49.0);
                        auto tmp14 = tmp12 / tmp13;
                        auto tmp15 = decltype(tmp11)(tmp11 + tmp14);
                        auto tmp18 = decltype(tmp16)(tmp16 - tmp17);
                        auto tmp20 = static_cast<float>(0.002551020408163265);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp23 = decltype(tmp22)(tmp22 * tmp22);
                        auto tmp24 = decltype(tmp21)(tmp21 * tmp23);
                        auto tmp25 = decltype(tmp18)(tmp18 * tmp24);
                        auto tmp26 = decltype(tmp15)(tmp15 - tmp25);
                        auto tmp28 = decltype(tmp27)(tmp27 * tmp20);
                        auto tmp29 = decltype(tmp26)(tmp26 - tmp28);
                        in_out_ptr0[static_cast<long>(x2 + (972L*x1) + (47628L*x0))] = tmp29;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(968L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(968L); x0<static_cast<long>(972L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    in_out_ptr1[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(968L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (972L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (972L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(968L); x1<static_cast<long>(972L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (972L*x0))];
                    auto tmp1 = in_ptr6[static_cast<long>(x1)];
                    auto tmp2 = in_ptr7[static_cast<long>(x1)];
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                    in_out_ptr0[static_cast<long>(x1 + (972L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(968L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (972L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (972L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (972L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(968L); x0<static_cast<long>(972L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (972L*x1))];
                        auto tmp1 = in_ptr1[static_cast<long>(x0 + (972L*x1))];
                        auto tmp3 = in_ptr2[static_cast<long>(x0 + (972L*x1))];
                        auto tmp4 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = decltype(tmp2)(tmp2 * tmp5);
                        tmp_acc0 = tmp_acc0 + tmp2;
                        tmp_acc1 = tmp_acc1 + tmp6;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(968L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(968L); x0<static_cast<long>(972L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    out_ptr2[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(968L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (972L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (972L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (972L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.002551020408163265);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (972L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(968L); x1<static_cast<long>(972L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (972L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (972L*x0))];
                    auto tmp3 = in_ptr2[static_cast<long>(x1 + (972L*x0))];
                    auto tmp4 = in_ptr3[static_cast<long>(x1)];
                    auto tmp6 = out_ptr1[static_cast<long>(x1)];
                    auto tmp9 = in_ptr4[static_cast<long>(x1)];
                    auto tmp14 = out_ptr0[static_cast<long>(x1)];
                    auto tmp17 = in_ptr5[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                    auto tmp7 = static_cast<float>(0.002551020408163265);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp10 = decltype(tmp9)(tmp9 * tmp9);
                    auto tmp11 = decltype(tmp8)(tmp8 * tmp10);
                    auto tmp12 = decltype(tmp5)(tmp5 * tmp11);
                    auto tmp13 = decltype(tmp2)(tmp2 - tmp12);
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp7);
                    auto tmp16 = decltype(tmp13)(tmp13 - tmp15);
                    auto tmp18 = decltype(tmp9)(tmp9 * tmp17);
                    auto tmp19 = decltype(tmp16)(tmp16 * tmp18);
                    in_out_ptr0[static_cast<long>(x1 + (972L*x0))] = tmp19;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_slice_backward_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(162L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp39 = in_ptr3[static_cast<long>(x0 + (162L*x1))];
                    auto tmp40 = in_ptr4[static_cast<long>(x0)];
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = static_cast<long>(151);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = in_ptr0[static_cast<long>(x0 + (185L*x1))];
                        auto tmp5 = in_ptr1[static_cast<long>(x0 + (174L*x1))];
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp7 = in_ptr2[static_cast<long>(x0 + (162L*x1))];
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        return tmp8;
                    }
                    ;
                    auto tmp9 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                    auto tmp10 = static_cast<float>(0.0);
                    auto tmp11 = tmp2 ? tmp9 : tmp10;
                    auto tmp12 = tmp0 < tmp1;
                    auto tmp13 = [&]
                    {
                        auto tmp14 = in_ptr0[static_cast<long>(x0 + (185L*x1))];
                        auto tmp15 = in_ptr1[static_cast<long>(x0 + (174L*x1))];
                        auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                        auto tmp17 = in_ptr2[static_cast<long>(x0 + (162L*x1))];
                        auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                        return tmp18;
                    }
                    ;
                    auto tmp19 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                    auto tmp20 = tmp12 ? tmp19 : tmp10;
                    auto tmp21 = decltype(tmp11)(tmp11 + tmp20);
                    auto tmp22 = [&]
                    {
                        auto tmp23 = in_ptr0[static_cast<long>(x0 + (185L*x1))];
                        auto tmp24 = in_ptr1[static_cast<long>(x0 + (174L*x1))];
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        auto tmp26 = in_ptr2[static_cast<long>(x0 + (162L*x1))];
                        auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                        return tmp27;
                    }
                    ;
                    auto tmp28 = tmp2 ? tmp22() : static_cast<decltype(tmp22())>(0.0);
                    auto tmp29 = tmp2 ? tmp28 : tmp10;
                    auto tmp30 = [&]
                    {
                        auto tmp31 = in_ptr0[static_cast<long>(x0 + (185L*x1))];
                        auto tmp32 = in_ptr1[static_cast<long>(x0 + (174L*x1))];
                        auto tmp33 = decltype(tmp31)(tmp31 + tmp32);
                        auto tmp34 = in_ptr2[static_cast<long>(x0 + (162L*x1))];
                        auto tmp35 = decltype(tmp33)(tmp33 + tmp34);
                        return tmp35;
                    }
                    ;
                    auto tmp36 = tmp12 ? tmp30() : static_cast<decltype(tmp30())>(0.0);
                    auto tmp37 = tmp12 ? tmp36 : tmp10;
                    auto tmp38 = decltype(tmp29)(tmp29 + tmp37);
                    auto tmp41 = decltype(tmp39)(tmp39 - tmp40);
                    auto tmp42 = decltype(tmp38)(tmp38 * tmp41);
                    tmp_acc0 = tmp_acc0 + tmp21;
                    tmp_acc1 = tmp_acc1 + tmp42;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(162L); x2+=static_cast<long>(1L))
                {
                    auto tmp22 = in_ptr3[static_cast<long>(x2 + (162L*x1) + (7938L*x0))];
                    auto tmp23 = in_ptr4[static_cast<long>(x2)];
                    auto tmp25 = out_ptr1[static_cast<long>(x2)];
                    auto tmp28 = in_ptr5[static_cast<long>(x2)];
                    auto tmp33 = out_ptr0[static_cast<long>(x2)];
                    auto tmp36 = in_ptr6[static_cast<long>(x2)];
                    auto tmp0 = c10::convert<long>(x2);
                    auto tmp1 = static_cast<long>(151);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = in_ptr0[static_cast<long>(x2 + (185L*x1) + (9065L*x0))];
                        auto tmp5 = in_ptr1[static_cast<long>(x2 + (174L*x1) + (8526L*x0))];
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp7 = in_ptr2[static_cast<long>(x2 + (162L*x1) + (7938L*x0))];
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        return tmp8;
                    }
                    ;
                    auto tmp9 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                    auto tmp10 = static_cast<float>(0.0);
                    auto tmp11 = tmp2 ? tmp9 : tmp10;
                    auto tmp12 = tmp0 < tmp1;
                    auto tmp13 = [&]
                    {
                        auto tmp14 = in_ptr0[static_cast<long>(x2 + (185L*x1) + (9065L*x0))];
                        auto tmp15 = in_ptr1[static_cast<long>(x2 + (174L*x1) + (8526L*x0))];
                        auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                        auto tmp17 = in_ptr2[static_cast<long>(x2 + (162L*x1) + (7938L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                        return tmp18;
                    }
                    ;
                    auto tmp19 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                    auto tmp20 = tmp12 ? tmp19 : tmp10;
                    auto tmp21 = decltype(tmp11)(tmp11 + tmp20);
                    auto tmp24 = decltype(tmp22)(tmp22 - tmp23);
                    auto tmp26 = static_cast<float>(0.002551020408163265);
                    auto tmp27 = decltype(tmp25)(tmp25 * tmp26);
                    auto tmp29 = decltype(tmp28)(tmp28 * tmp28);
                    auto tmp30 = decltype(tmp27)(tmp27 * tmp29);
                    auto tmp31 = decltype(tmp24)(tmp24 * tmp30);
                    auto tmp32 = decltype(tmp21)(tmp21 - tmp31);
                    auto tmp34 = decltype(tmp33)(tmp33 * tmp26);
                    auto tmp35 = decltype(tmp32)(tmp32 - tmp34);
                    auto tmp37 = decltype(tmp28)(tmp28 * tmp36);
                    auto tmp38 = decltype(tmp35)(tmp35 * tmp37);
                    out_ptr2[static_cast<long>(x1 + (49L*x2) + (7938L*x0))] = tmp38;
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(160L); x0<static_cast<long>(162L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr5[static_cast<long>(x0)];
            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
            in_out_ptr0[static_cast<long>(x0)] = tmp2;
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(906L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (906L*x2) + (44394L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (906L*x0))];
                            auto tmp9 = in_ptr2[static_cast<long>(x1 + (906L*x2) + (44394L*x0))];
                            auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = tmp3 <= tmp4;
                            auto tmp6 = static_cast<float>(6.0);
                            auto tmp7 = tmp3 >= tmp6;
                            auto tmp8 = decltype(tmp5)(tmp5 | tmp7);
                            auto tmp10 = tmp8 ? tmp4 : tmp9;
                            auto tmp11 = decltype(tmp10)(tmp10 * tmp0);
                            tmp_acc0 = tmp_acc0 + tmp11;
                        }
                        out_ptr0[static_cast<long>(x1 + (906L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(7248L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_threshold_backward_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (75L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (75L*x1)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (75L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp9 = tmp5 * tmp8;
                    tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    tmp_acc1_vec = tmp_acc1_vec + tmp9;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(72L); x0<static_cast<long>(75L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (75L*x1))];
                    auto tmp3 = in_ptr1[static_cast<long>(x0 + (75L*x1))];
                    auto tmp5 = in_ptr2[static_cast<long>(x0 + (75L*x1))];
                    auto tmp6 = in_ptr3[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    auto tmp4 = tmp2 ? tmp1 : tmp3;
                    auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                    auto tmp8 = decltype(tmp4)(tmp4 * tmp7);
                    tmp_acc0 = tmp_acc0 + tmp4;
                    tmp_acc1 = tmp_acc1 + tmp8;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(8.0);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 / tmp3;
            auto tmp5 = static_cast<float>(1e-05);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 + tmp6;
            auto tmp8 = tmp7.rsqrt();
            auto tmp9 = tmp0 * tmp8;
            tmp9.store(out_ptr2 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(72L); x0<static_cast<long>(75L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr4[static_cast<long>(x0)];
            auto tmp2 = static_cast<float>(8.0);
            auto tmp3 = tmp1 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
            auto tmp6 = 1 / std::sqrt(tmp5);
            auto tmp7 = decltype(tmp0)(tmp0 * tmp6);
            out_ptr2[static_cast<long>(x0)] = tmp7;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (75L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (75L*x0)));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (75L*x0)));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp25 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                auto tmp8 = tmp6 - tmp7;
                auto tmp10 = static_cast<float>(0.125);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp14 = static_cast<float>(8.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 / tmp15;
                auto tmp17 = static_cast<float>(1e-05);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 + tmp18;
                auto tmp20 = tmp19.rsqrt();
                auto tmp21 = tmp20 * tmp20;
                auto tmp22 = tmp12 * tmp21;
                auto tmp23 = tmp8 * tmp22;
                auto tmp24 = tmp5 - tmp23;
                auto tmp26 = tmp25 * tmp11;
                auto tmp27 = tmp24 - tmp26;
                auto tmp29 = tmp20 * tmp28;
                auto tmp30 = tmp27 * tmp29;
                tmp30.store(in_out_ptr0 + static_cast<long>(x1 + (75L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(72L); x1<static_cast<long>(75L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (75L*x0))];
                auto tmp3 = in_out_ptr0[static_cast<long>(x1 + (75L*x0))];
                auto tmp5 = in_ptr2[static_cast<long>(x1 + (75L*x0))];
                auto tmp6 = in_ptr3[static_cast<long>(x1)];
                auto tmp8 = out_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr4[static_cast<long>(x1)];
                auto tmp21 = out_ptr0[static_cast<long>(x1)];
                auto tmp24 = in_ptr5[static_cast<long>(x1)];
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp4 = tmp2 ? tmp1 : tmp3;
                auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                auto tmp9 = static_cast<float>(0.125);
                auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                auto tmp12 = static_cast<float>(8.0);
                auto tmp13 = tmp11 / tmp12;
                auto tmp14 = static_cast<float>(1e-05);
                auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                auto tmp16 = 1 / std::sqrt(tmp15);
                auto tmp17 = decltype(tmp16)(tmp16 * tmp16);
                auto tmp18 = decltype(tmp10)(tmp10 * tmp17);
                auto tmp19 = decltype(tmp7)(tmp7 * tmp18);
                auto tmp20 = decltype(tmp4)(tmp4 - tmp19);
                auto tmp22 = decltype(tmp21)(tmp21 * tmp9);
                auto tmp23 = decltype(tmp20)(tmp20 - tmp22);
                auto tmp25 = decltype(tmp16)(tmp16 * tmp24);
                auto tmp26 = decltype(tmp23)(tmp23 * tmp25);
                in_out_ptr0[static_cast<long>(x1 + (75L*x0))] = tmp26;
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0)
{
    auto in_ptr2 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(906L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x0 + (906L*x2) + (44394L*x1))];
                            auto tmp1 = in_ptr1[static_cast<long>(x0 + (906L*x1))];
                            auto tmp9 = in_ptr2[static_cast<long>(x0 + (906L*x2) + (44394L*x1))];
                            auto tmp12 = in_ptr3[static_cast<long>(x0 + (906L*x1))];
                            auto tmp16 = in_ptr4[static_cast<long>(x0 + (906L*x2) + (44394L*x1))];
                            auto tmp17 = in_ptr5[static_cast<long>(x0)];
                            auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = tmp3 <= tmp4;
                            auto tmp6 = static_cast<float>(6.0);
                            auto tmp7 = tmp3 >= tmp6;
                            auto tmp8 = decltype(tmp5)(tmp5 | tmp7);
                            auto tmp10 = tmp8 ? tmp4 : tmp9;
                            auto tmp11 = decltype(tmp10)(tmp10 * tmp2);
                            auto tmp13 = static_cast<float>(49.0);
                            auto tmp14 = tmp12 / tmp13;
                            auto tmp15 = decltype(tmp11)(tmp11 + tmp14);
                            auto tmp18 = decltype(tmp16)(tmp16 - tmp17);
                            auto tmp19 = decltype(tmp15)(tmp15 * tmp18);
                            tmp_acc0 = tmp_acc0 + tmp15;
                            tmp_acc1 = tmp_acc1 + tmp19;
                        }
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(906L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (906L*x1) + (44394L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (906L*x0))];
                        auto tmp9 = in_out_ptr0[static_cast<long>(x2 + (906L*x1) + (44394L*x0))];
                        auto tmp12 = in_ptr3[static_cast<long>(x2 + (906L*x0))];
                        auto tmp16 = in_ptr4[static_cast<long>(x2 + (906L*x1) + (44394L*x0))];
                        auto tmp17 = in_ptr5[static_cast<long>(x2)];
                        auto tmp19 = out_ptr1[static_cast<long>(x2)];
                        auto tmp22 = in_ptr6[static_cast<long>(x2)];
                        auto tmp27 = out_ptr0[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                        auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = tmp3 <= tmp4;
                        auto tmp6 = static_cast<float>(6.0);
                        auto tmp7 = tmp3 >= tmp6;
                        auto tmp8 = decltype(tmp5)(tmp5 | tmp7);
                        auto tmp10 = tmp8 ? tmp4 : tmp9;
                        auto tmp11 = decltype(tmp10)(tmp10 * tmp2);
                        auto tmp13 = static_cast<float>(49.0);
                        auto tmp14 = tmp12 / tmp13;
                        auto tmp15 = decltype(tmp11)(tmp11 + tmp14);
                        auto tmp18 = decltype(tmp16)(tmp16 - tmp17);
                        auto tmp20 = static_cast<float>(0.002551020408163265);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp23 = decltype(tmp22)(tmp22 * tmp22);
                        auto tmp24 = decltype(tmp21)(tmp21 * tmp23);
                        auto tmp25 = decltype(tmp18)(tmp18 * tmp24);
                        auto tmp26 = decltype(tmp15)(tmp15 - tmp25);
                        auto tmp28 = decltype(tmp27)(tmp27 * tmp20);
                        auto tmp29 = decltype(tmp26)(tmp26 - tmp28);
                        in_out_ptr0[static_cast<long>(x2 + (906L*x1) + (44394L*x0))] = tmp29;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(904L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(904L); x0<static_cast<long>(906L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    in_out_ptr1[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(904L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (906L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (906L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(904L); x1<static_cast<long>(906L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (906L*x0))];
                    auto tmp1 = in_ptr6[static_cast<long>(x1)];
                    auto tmp2 = in_ptr7[static_cast<long>(x1)];
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                    in_out_ptr0[static_cast<long>(x1 + (906L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(904L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (906L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (906L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (906L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(904L); x0<static_cast<long>(906L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (906L*x1))];
                        auto tmp1 = in_ptr1[static_cast<long>(x0 + (906L*x1))];
                        auto tmp3 = in_ptr2[static_cast<long>(x0 + (906L*x1))];
                        auto tmp4 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = decltype(tmp2)(tmp2 * tmp5);
                        tmp_acc0 = tmp_acc0 + tmp2;
                        tmp_acc1 = tmp_acc1 + tmp6;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(904L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(904L); x0<static_cast<long>(906L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    out_ptr2[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(904L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (906L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (906L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (906L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.002551020408163265);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (906L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(904L); x1<static_cast<long>(906L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (906L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (906L*x0))];
                    auto tmp3 = in_ptr2[static_cast<long>(x1 + (906L*x0))];
                    auto tmp4 = in_ptr3[static_cast<long>(x1)];
                    auto tmp6 = out_ptr1[static_cast<long>(x1)];
                    auto tmp9 = in_ptr4[static_cast<long>(x1)];
                    auto tmp14 = out_ptr0[static_cast<long>(x1)];
                    auto tmp17 = in_ptr5[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                    auto tmp7 = static_cast<float>(0.002551020408163265);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp10 = decltype(tmp9)(tmp9 * tmp9);
                    auto tmp11 = decltype(tmp8)(tmp8 * tmp10);
                    auto tmp12 = decltype(tmp5)(tmp5 * tmp11);
                    auto tmp13 = decltype(tmp2)(tmp2 - tmp12);
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp7);
                    auto tmp16 = decltype(tmp13)(tmp13 - tmp15);
                    auto tmp18 = decltype(tmp9)(tmp9 * tmp17);
                    auto tmp19 = decltype(tmp16)(tmp16 * tmp18);
                    in_out_ptr0[static_cast<long>(x1 + (906L*x0))] = tmp19;
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(151L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp47 = in_ptr4[static_cast<long>(x0 + (151L*x1))];
                    auto tmp48 = in_ptr5[static_cast<long>(x0)];
                    auto tmp0 = c10::convert<long>(x0);
                    auto tmp1 = static_cast<long>(140);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = in_ptr0[static_cast<long>(x0 + (185L*x1))];
                        auto tmp5 = in_ptr1[static_cast<long>(x0 + (174L*x1))];
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp7 = in_ptr2[static_cast<long>(x0 + (162L*x1))];
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = in_ptr3[static_cast<long>(x0 + (151L*x1))];
                        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                    auto tmp12 = static_cast<float>(0.0);
                    auto tmp13 = tmp2 ? tmp11 : tmp12;
                    auto tmp14 = tmp0 < tmp1;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr0[static_cast<long>(x0 + (185L*x1))];
                        auto tmp17 = in_ptr1[static_cast<long>(x0 + (174L*x1))];
                        auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>(x0 + (162L*x1))];
                        auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                        auto tmp21 = in_ptr3[static_cast<long>(x0 + (151L*x1))];
                        auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                        return tmp22;
                    }
                    ;
                    auto tmp23 = tmp14 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp24 = tmp14 ? tmp23 : tmp12;
                    auto tmp25 = decltype(tmp13)(tmp13 + tmp24);
                    auto tmp26 = [&]
                    {
                        auto tmp27 = in_ptr0[static_cast<long>(x0 + (185L*x1))];
                        auto tmp28 = in_ptr1[static_cast<long>(x0 + (174L*x1))];
                        auto tmp29 = decltype(tmp27)(tmp27 + tmp28);
                        auto tmp30 = in_ptr2[static_cast<long>(x0 + (162L*x1))];
                        auto tmp31 = decltype(tmp29)(tmp29 + tmp30);
                        auto tmp32 = in_ptr3[static_cast<long>(x0 + (151L*x1))];
                        auto tmp33 = decltype(tmp31)(tmp31 + tmp32);
                        return tmp33;
                    }
                    ;
                    auto tmp34 = tmp2 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                    auto tmp35 = tmp2 ? tmp34 : tmp12;
                    auto tmp36 = [&]
                    {
                        auto tmp37 = in_ptr0[static_cast<long>(x0 + (185L*x1))];
                        auto tmp38 = in_ptr1[static_cast<long>(x0 + (174L*x1))];
                        auto tmp39 = decltype(tmp37)(tmp37 + tmp38);
                        auto tmp40 = in_ptr2[static_cast<long>(x0 + (162L*x1))];
                        auto tmp41 = decltype(tmp39)(tmp39 + tmp40);
                        auto tmp42 = in_ptr3[static_cast<long>(x0 + (151L*x1))];
                        auto tmp43 = decltype(tmp41)(tmp41 + tmp42);
                        return tmp43;
                    }
                    ;
                    auto tmp44 = tmp14 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                    auto tmp45 = tmp14 ? tmp44 : tmp12;
                    auto tmp46 = decltype(tmp35)(tmp35 + tmp45);
                    auto tmp49 = decltype(tmp47)(tmp47 - tmp48);
                    auto tmp50 = decltype(tmp46)(tmp46 * tmp49);
                    tmp_acc0 = tmp_acc0 + tmp25;
                    tmp_acc1 = tmp_acc1 + tmp50;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(151L); x1+=static_cast<long>(1L))
            {
                auto tmp26 = in_ptr4[static_cast<long>(x1 + (151L*x0))];
                auto tmp27 = in_ptr5[static_cast<long>(x1)];
                auto tmp29 = out_ptr1[static_cast<long>(x1)];
                auto tmp32 = in_ptr6[static_cast<long>(x1)];
                auto tmp37 = out_ptr0[static_cast<long>(x1)];
                auto tmp0 = c10::convert<long>(x1);
                auto tmp1 = static_cast<long>(140);
                auto tmp2 = tmp0 >= tmp1;
                auto tmp3 = [&]
                {
                    auto tmp4 = in_ptr0[static_cast<long>(x1 + (185L*x0))];
                    auto tmp5 = in_ptr1[static_cast<long>(x1 + (174L*x0))];
                    auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                    auto tmp7 = in_ptr2[static_cast<long>(x1 + (162L*x0))];
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = in_ptr3[static_cast<long>(x1 + (151L*x0))];
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    return tmp10;
                }
                ;
                auto tmp11 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                auto tmp12 = static_cast<float>(0.0);
                auto tmp13 = tmp2 ? tmp11 : tmp12;
                auto tmp14 = tmp0 < tmp1;
                auto tmp15 = [&]
                {
                    auto tmp16 = in_ptr0[static_cast<long>(x1 + (185L*x0))];
                    auto tmp17 = in_ptr1[static_cast<long>(x1 + (174L*x0))];
                    auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                    auto tmp19 = in_ptr2[static_cast<long>(x1 + (162L*x0))];
                    auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                    auto tmp21 = in_ptr3[static_cast<long>(x1 + (151L*x0))];
                    auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                    return tmp22;
                }
                ;
                auto tmp23 = tmp14 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                auto tmp24 = tmp14 ? tmp23 : tmp12;
                auto tmp25 = decltype(tmp13)(tmp13 + tmp24);
                auto tmp28 = decltype(tmp26)(tmp26 - tmp27);
                auto tmp30 = static_cast<float>(0.002551020408163265);
                auto tmp31 = decltype(tmp29)(tmp29 * tmp30);
                auto tmp33 = decltype(tmp32)(tmp32 * tmp32);
                auto tmp34 = decltype(tmp31)(tmp31 * tmp33);
                auto tmp35 = decltype(tmp28)(tmp28 * tmp34);
                auto tmp36 = decltype(tmp25)(tmp25 - tmp35);
                auto tmp38 = decltype(tmp37)(tmp37 * tmp30);
                auto tmp39 = decltype(tmp36)(tmp36 - tmp38);
                out_ptr2[static_cast<long>(x1 + (151L*x0))] = tmp39;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(144L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(144L); x0<static_cast<long>(151L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr6[static_cast<long>(x0)];
            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
            in_out_ptr0[static_cast<long>(x0)] = tmp2;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                {
                    float tmp0[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(x1 + (151L*x2) + (7399L*x0)), static_cast<long>(151L), tmp0, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                        auto tmp2 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                        auto tmp3 = in_ptr7[static_cast<long>(x1 + x1_inner)];
                        auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp1 * tmp5;
                        tmp6.store(out_ptr3 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (7399L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (151L*x2) + (7399L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp4.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr3[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (7399L*x0))] = tmpbuf[x1_inner]; }
                }
            }
            #pragma GCC ivdep
            for(long x1=static_cast<long>(144L); x1<static_cast<long>(151L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>(x1 + (151L*x2) + (7399L*x0))];
                    auto tmp1 = in_ptr6[static_cast<long>(x1)];
                    auto tmp2 = in_ptr7[static_cast<long>(x1)];
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                    out_ptr3[static_cast<long>(x2 + (49L*x1) + (7399L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(840L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (840L*x2) + (41160L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (840L*x0))];
                            auto tmp9 = in_ptr2[static_cast<long>(x1 + (840L*x2) + (41160L*x0))];
                            auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = tmp3 <= tmp4;
                            auto tmp6 = static_cast<float>(6.0);
                            auto tmp7 = tmp3 >= tmp6;
                            auto tmp8 = decltype(tmp5)(tmp5 | tmp7);
                            auto tmp10 = tmp8 ? tmp4 : tmp9;
                            auto tmp11 = decltype(tmp10)(tmp10 * tmp0);
                            tmp_acc0 = tmp_acc0 + tmp11;
                        }
                        out_ptr0[static_cast<long>(x1 + (840L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(6720L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_threshold_backward_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (70L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (70L*x1)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (70L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp9 = tmp5 * tmp8;
                    tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    tmp_acc1_vec = tmp_acc1_vec + tmp9;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(64L); x0<static_cast<long>(70L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (70L*x1))];
                    auto tmp3 = in_ptr1[static_cast<long>(x0 + (70L*x1))];
                    auto tmp5 = in_ptr2[static_cast<long>(x0 + (70L*x1))];
                    auto tmp6 = in_ptr3[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    auto tmp4 = tmp2 ? tmp1 : tmp3;
                    auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                    auto tmp8 = decltype(tmp4)(tmp4 * tmp7);
                    tmp_acc0 = tmp_acc0 + tmp4;
                    tmp_acc1 = tmp_acc1 + tmp8;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(8.0);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 / tmp3;
            auto tmp5 = static_cast<float>(1e-05);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 + tmp6;
            auto tmp8 = tmp7.rsqrt();
            auto tmp9 = tmp0 * tmp8;
            tmp9.store(out_ptr2 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(64L); x0<static_cast<long>(70L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr4[static_cast<long>(x0)];
            auto tmp2 = static_cast<float>(8.0);
            auto tmp3 = tmp1 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
            auto tmp6 = 1 / std::sqrt(tmp5);
            auto tmp7 = decltype(tmp0)(tmp0 * tmp6);
            out_ptr2[static_cast<long>(x0)] = tmp7;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (70L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (70L*x0)));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (70L*x0)));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp25 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                auto tmp8 = tmp6 - tmp7;
                auto tmp10 = static_cast<float>(0.125);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp14 = static_cast<float>(8.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 / tmp15;
                auto tmp17 = static_cast<float>(1e-05);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 + tmp18;
                auto tmp20 = tmp19.rsqrt();
                auto tmp21 = tmp20 * tmp20;
                auto tmp22 = tmp12 * tmp21;
                auto tmp23 = tmp8 * tmp22;
                auto tmp24 = tmp5 - tmp23;
                auto tmp26 = tmp25 * tmp11;
                auto tmp27 = tmp24 - tmp26;
                auto tmp29 = tmp20 * tmp28;
                auto tmp30 = tmp27 * tmp29;
                tmp30.store(in_out_ptr0 + static_cast<long>(x1 + (70L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(64L); x1<static_cast<long>(70L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (70L*x0))];
                auto tmp3 = in_out_ptr0[static_cast<long>(x1 + (70L*x0))];
                auto tmp5 = in_ptr2[static_cast<long>(x1 + (70L*x0))];
                auto tmp6 = in_ptr3[static_cast<long>(x1)];
                auto tmp8 = out_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr4[static_cast<long>(x1)];
                auto tmp21 = out_ptr0[static_cast<long>(x1)];
                auto tmp24 = in_ptr5[static_cast<long>(x1)];
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp4 = tmp2 ? tmp1 : tmp3;
                auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                auto tmp9 = static_cast<float>(0.125);
                auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                auto tmp12 = static_cast<float>(8.0);
                auto tmp13 = tmp11 / tmp12;
                auto tmp14 = static_cast<float>(1e-05);
                auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                auto tmp16 = 1 / std::sqrt(tmp15);
                auto tmp17 = decltype(tmp16)(tmp16 * tmp16);
                auto tmp18 = decltype(tmp10)(tmp10 * tmp17);
                auto tmp19 = decltype(tmp7)(tmp7 * tmp18);
                auto tmp20 = decltype(tmp4)(tmp4 - tmp19);
                auto tmp22 = decltype(tmp21)(tmp21 * tmp9);
                auto tmp23 = decltype(tmp20)(tmp20 - tmp22);
                auto tmp25 = decltype(tmp16)(tmp16 * tmp24);
                auto tmp26 = decltype(tmp23)(tmp23 * tmp25);
                in_out_ptr0[static_cast<long>(x1 + (70L*x0))] = tmp26;
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0)
{
    auto in_ptr2 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(840L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x0 + (840L*x2) + (41160L*x1))];
                            auto tmp1 = in_ptr1[static_cast<long>(x0 + (840L*x1))];
                            auto tmp9 = in_ptr2[static_cast<long>(x0 + (840L*x2) + (41160L*x1))];
                            auto tmp12 = in_ptr3[static_cast<long>(x0 + (840L*x1))];
                            auto tmp16 = in_ptr4[static_cast<long>(x0 + (840L*x2) + (41160L*x1))];
                            auto tmp17 = in_ptr5[static_cast<long>(x0)];
                            auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = tmp3 <= tmp4;
                            auto tmp6 = static_cast<float>(6.0);
                            auto tmp7 = tmp3 >= tmp6;
                            auto tmp8 = decltype(tmp5)(tmp5 | tmp7);
                            auto tmp10 = tmp8 ? tmp4 : tmp9;
                            auto tmp11 = decltype(tmp10)(tmp10 * tmp2);
                            auto tmp13 = static_cast<float>(49.0);
                            auto tmp14 = tmp12 / tmp13;
                            auto tmp15 = decltype(tmp11)(tmp11 + tmp14);
                            auto tmp18 = decltype(tmp16)(tmp16 - tmp17);
                            auto tmp19 = decltype(tmp15)(tmp15 * tmp18);
                            tmp_acc0 = tmp_acc0 + tmp15;
                            tmp_acc1 = tmp_acc1 + tmp19;
                        }
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(840L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (840L*x1) + (41160L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (840L*x0))];
                        auto tmp9 = in_out_ptr0[static_cast<long>(x2 + (840L*x1) + (41160L*x0))];
                        auto tmp12 = in_ptr3[static_cast<long>(x2 + (840L*x0))];
                        auto tmp16 = in_ptr4[static_cast<long>(x2 + (840L*x1) + (41160L*x0))];
                        auto tmp17 = in_ptr5[static_cast<long>(x2)];
                        auto tmp19 = out_ptr1[static_cast<long>(x2)];
                        auto tmp22 = in_ptr6[static_cast<long>(x2)];
                        auto tmp27 = out_ptr0[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                        auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = tmp3 <= tmp4;
                        auto tmp6 = static_cast<float>(6.0);
                        auto tmp7 = tmp3 >= tmp6;
                        auto tmp8 = decltype(tmp5)(tmp5 | tmp7);
                        auto tmp10 = tmp8 ? tmp4 : tmp9;
                        auto tmp11 = decltype(tmp10)(tmp10 * tmp2);
                        auto tmp13 = static_cast<float>(49.0);
                        auto tmp14 = tmp12 / tmp13;
                        auto tmp15 = decltype(tmp11)(tmp11 + tmp14);
                        auto tmp18 = decltype(tmp16)(tmp16 - tmp17);
                        auto tmp20 = static_cast<float>(0.002551020408163265);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp23 = decltype(tmp22)(tmp22 * tmp22);
                        auto tmp24 = decltype(tmp21)(tmp21 * tmp23);
                        auto tmp25 = decltype(tmp18)(tmp18 * tmp24);
                        auto tmp26 = decltype(tmp15)(tmp15 - tmp25);
                        auto tmp28 = decltype(tmp27)(tmp27 * tmp20);
                        auto tmp29 = decltype(tmp26)(tmp26 - tmp28);
                        in_out_ptr0[static_cast<long>(x2 + (840L*x1) + (41160L*x0))] = tmp29;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(840L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(840L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (840L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (840L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(840L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (840L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (840L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (840L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(840L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(840L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (840L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (840L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (840L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.002551020408163265);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (840L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
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
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(136L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (185L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (174L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (162L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (151L*x0)));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (140L*x0)));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (140L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(136L); x1<static_cast<long>(140L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (185L*x0))];
                auto tmp1 = in_ptr1[static_cast<long>(x1 + (174L*x0))];
                auto tmp3 = in_ptr2[static_cast<long>(x1 + (162L*x0))];
                auto tmp5 = in_ptr3[static_cast<long>(x1 + (151L*x0))];
                auto tmp7 = in_out_ptr0[static_cast<long>(x1 + (140L*x0))];
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                in_out_ptr0[static_cast<long>(x1 + (140L*x0))] = tmp8;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(136L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (140L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (140L*x1)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    tmp_acc1_vec = tmp_acc1_vec + tmp4;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(136L); x0<static_cast<long>(140L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x0 + (140L*x1))];
                    auto tmp1 = in_ptr4[static_cast<long>(x0 + (140L*x1))];
                    auto tmp2 = in_ptr5[static_cast<long>(x0)];
                    auto tmp3 = decltype(tmp1)(tmp1 - tmp2);
                    auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                    tmp_acc0 = tmp_acc0 + tmp0;
                    tmp_acc1 = tmp_acc1 + tmp4;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(136L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(136L); x0<static_cast<long>(140L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr6[static_cast<long>(x0)];
            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
            out_ptr2[static_cast<long>(x0)] = tmp2;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(136L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (140L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (140L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(0.002551020408163265);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp9 = tmp8 * tmp8;
                auto tmp10 = tmp7 * tmp9;
                auto tmp11 = tmp3 * tmp10;
                auto tmp12 = tmp0 - tmp11;
                auto tmp14 = tmp13 * tmp6;
                auto tmp15 = tmp12 - tmp14;
                auto tmp17 = tmp8 * tmp16;
                auto tmp18 = tmp15 * tmp17;
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (140L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(136L); x1<static_cast<long>(140L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (140L*x0))];
                auto tmp1 = in_ptr4[static_cast<long>(x1 + (140L*x0))];
                auto tmp2 = in_ptr5[static_cast<long>(x1)];
                auto tmp4 = out_ptr1[static_cast<long>(x1)];
                auto tmp7 = in_ptr6[static_cast<long>(x1)];
                auto tmp12 = out_ptr0[static_cast<long>(x1)];
                auto tmp15 = in_ptr7[static_cast<long>(x1)];
                auto tmp3 = decltype(tmp1)(tmp1 - tmp2);
                auto tmp5 = static_cast<float>(0.002551020408163265);
                auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                auto tmp8 = decltype(tmp7)(tmp7 * tmp7);
                auto tmp9 = decltype(tmp6)(tmp6 * tmp8);
                auto tmp10 = decltype(tmp3)(tmp3 * tmp9);
                auto tmp11 = decltype(tmp0)(tmp0 - tmp10);
                auto tmp13 = decltype(tmp12)(tmp12 * tmp5);
                auto tmp14 = decltype(tmp11)(tmp11 - tmp13);
                auto tmp16 = decltype(tmp7)(tmp7 * tmp15);
                auto tmp17 = decltype(tmp14)(tmp14 * tmp16);
                in_out_ptr0[static_cast<long>(x1 + (140L*x0))] = tmp17;
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (768L*x2) + (37632L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (768L*x0))];
                            auto tmp9 = in_ptr2[static_cast<long>(x1 + (768L*x2) + (37632L*x0))];
                            auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = tmp3 <= tmp4;
                            auto tmp6 = static_cast<float>(6.0);
                            auto tmp7 = tmp3 >= tmp6;
                            auto tmp8 = decltype(tmp5)(tmp5 | tmp7);
                            auto tmp10 = tmp8 ? tmp4 : tmp9;
                            auto tmp11 = decltype(tmp10)(tmp10 * tmp0);
                            tmp_acc0 = tmp_acc0 + tmp11;
                        }
                        out_ptr0[static_cast<long>(x1 + (768L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_threshold_backward_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp9 = tmp5 * tmp8;
                    tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    tmp_acc1_vec = tmp_acc1_vec + tmp9;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(8.0);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 / tmp3;
            auto tmp5 = static_cast<float>(1e-05);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 + tmp6;
            auto tmp8 = tmp7.rsqrt();
            auto tmp9 = tmp0 * tmp8;
            tmp9.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (64L*x0)));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp25 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                auto tmp8 = tmp6 - tmp7;
                auto tmp10 = static_cast<float>(0.125);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp14 = static_cast<float>(8.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 / tmp15;
                auto tmp17 = static_cast<float>(1e-05);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 + tmp18;
                auto tmp20 = tmp19.rsqrt();
                auto tmp21 = tmp20 * tmp20;
                auto tmp22 = tmp12 * tmp21;
                auto tmp23 = tmp8 * tmp22;
                auto tmp24 = tmp5 - tmp23;
                auto tmp26 = tmp25 * tmp11;
                auto tmp27 = tmp24 - tmp26;
                auto tmp29 = tmp20 * tmp28;
                auto tmp30 = tmp27 * tmp29;
                tmp30.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0)
{
    auto in_ptr2 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x0 + (768L*x2) + (37632L*x1))];
                            auto tmp1 = in_ptr1[static_cast<long>(x0 + (768L*x1))];
                            auto tmp9 = in_ptr2[static_cast<long>(x0 + (768L*x2) + (37632L*x1))];
                            auto tmp12 = in_ptr3[static_cast<long>(x0 + (768L*x1))];
                            auto tmp16 = in_ptr4[static_cast<long>(x0 + (768L*x2) + (37632L*x1))];
                            auto tmp17 = in_ptr5[static_cast<long>(x0)];
                            auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = tmp3 <= tmp4;
                            auto tmp6 = static_cast<float>(6.0);
                            auto tmp7 = tmp3 >= tmp6;
                            auto tmp8 = decltype(tmp5)(tmp5 | tmp7);
                            auto tmp10 = tmp8 ? tmp4 : tmp9;
                            auto tmp11 = decltype(tmp10)(tmp10 * tmp2);
                            auto tmp13 = static_cast<float>(49.0);
                            auto tmp14 = tmp12 / tmp13;
                            auto tmp15 = decltype(tmp11)(tmp11 + tmp14);
                            auto tmp18 = decltype(tmp16)(tmp16 - tmp17);
                            auto tmp19 = decltype(tmp15)(tmp15 * tmp18);
                            tmp_acc0 = tmp_acc0 + tmp15;
                            tmp_acc1 = tmp_acc1 + tmp19;
                        }
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (768L*x1) + (37632L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (768L*x0))];
                        auto tmp9 = in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (37632L*x0))];
                        auto tmp12 = in_ptr3[static_cast<long>(x2 + (768L*x0))];
                        auto tmp16 = in_ptr4[static_cast<long>(x2 + (768L*x1) + (37632L*x0))];
                        auto tmp17 = in_ptr5[static_cast<long>(x2)];
                        auto tmp19 = out_ptr1[static_cast<long>(x2)];
                        auto tmp22 = in_ptr6[static_cast<long>(x2)];
                        auto tmp27 = out_ptr0[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                        auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = tmp3 <= tmp4;
                        auto tmp6 = static_cast<float>(6.0);
                        auto tmp7 = tmp3 >= tmp6;
                        auto tmp8 = decltype(tmp5)(tmp5 | tmp7);
                        auto tmp10 = tmp8 ? tmp4 : tmp9;
                        auto tmp11 = decltype(tmp10)(tmp10 * tmp2);
                        auto tmp13 = static_cast<float>(49.0);
                        auto tmp14 = tmp12 / tmp13;
                        auto tmp15 = decltype(tmp11)(tmp11 + tmp14);
                        auto tmp18 = decltype(tmp16)(tmp16 - tmp17);
                        auto tmp20 = static_cast<float>(0.002551020408163265);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp23 = decltype(tmp22)(tmp22 * tmp22);
                        auto tmp24 = decltype(tmp21)(tmp21 * tmp23);
                        auto tmp25 = decltype(tmp18)(tmp18 * tmp24);
                        auto tmp26 = decltype(tmp15)(tmp15 - tmp25);
                        auto tmp28 = decltype(tmp27)(tmp27 * tmp20);
                        auto tmp29 = decltype(tmp26)(tmp26 - tmp28);
                        in_out_ptr0[static_cast<long>(x2 + (768L*x1) + (37632L*x0))] = tmp29;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.0006377551020408163);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_27 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp23 = in_ptr1[static_cast<long>(x0 + (128L*x1))];
                        auto tmp24 = in_ptr2[static_cast<long>(x0)];
                        auto tmp0 = c10::convert<long>(x0);
                        auto tmp1 = static_cast<long>(117);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr0[static_cast<long>(x0 + (128L*x1))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = tmp2 ? tmp5 : tmp6;
                        auto tmp8 = tmp0 < tmp1;
                        auto tmp9 = [&]
                        {
                            auto tmp10 = in_ptr0[static_cast<long>(x0 + (128L*x1))];
                            return tmp10;
                        }
                        ;
                        auto tmp11 = tmp8 ? tmp9() : static_cast<decltype(tmp9())>(0.0);
                        auto tmp12 = tmp8 ? tmp11 : tmp6;
                        auto tmp13 = decltype(tmp7)(tmp7 + tmp12);
                        auto tmp14 = [&]
                        {
                            auto tmp15 = in_ptr0[static_cast<long>(x0 + (128L*x1))];
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp2 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                        auto tmp17 = tmp2 ? tmp16 : tmp6;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = in_ptr0[static_cast<long>(x0 + (128L*x1))];
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp8 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                        auto tmp21 = tmp8 ? tmp20 : tmp6;
                        auto tmp22 = decltype(tmp17)(tmp17 + tmp21);
                        auto tmp25 = decltype(tmp23)(tmp23 - tmp24);
                        auto tmp26 = decltype(tmp22)(tmp22 * tmp25);
                        tmp_acc0 = tmp_acc0 + tmp13;
                        tmp_acc1 = tmp_acc1 + tmp26;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp14 = in_ptr1[static_cast<long>(x2 + (128L*x1) + (25088L*x0))];
                        auto tmp15 = in_ptr2[static_cast<long>(x2)];
                        auto tmp17 = out_ptr1[static_cast<long>(x2)];
                        auto tmp20 = in_ptr3[static_cast<long>(x2)];
                        auto tmp25 = out_ptr0[static_cast<long>(x2)];
                        auto tmp28 = in_ptr4[static_cast<long>(x2)];
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(117);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (25088L*x0))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = tmp2 ? tmp5 : tmp6;
                        auto tmp8 = tmp0 < tmp1;
                        auto tmp9 = [&]
                        {
                            auto tmp10 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (25088L*x0))];
                            return tmp10;
                        }
                        ;
                        auto tmp11 = tmp8 ? tmp9() : static_cast<decltype(tmp9())>(0.0);
                        auto tmp12 = tmp8 ? tmp11 : tmp6;
                        auto tmp13 = decltype(tmp7)(tmp7 + tmp12);
                        auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                        auto tmp18 = static_cast<float>(0.0006377551020408163);
                        auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                        auto tmp21 = decltype(tmp20)(tmp20 * tmp20);
                        auto tmp22 = decltype(tmp19)(tmp19 * tmp21);
                        auto tmp23 = decltype(tmp16)(tmp16 * tmp22);
                        auto tmp24 = decltype(tmp13)(tmp13 - tmp23);
                        auto tmp26 = decltype(tmp25)(tmp25 * tmp18);
                        auto tmp27 = decltype(tmp24)(tmp24 - tmp26);
                        auto tmp29 = decltype(tmp20)(tmp20 * tmp28);
                        auto tmp30 = decltype(tmp27)(tmp27 * tmp29);
                        out_ptr3[static_cast<long>(x1 + (196L*x2) + (25088L*x0))] = tmp30;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(702L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (702L*x2) + (137592L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (702L*x0))];
                            auto tmp9 = in_ptr2[static_cast<long>(x1 + (702L*x2) + (137592L*x0))];
                            auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = tmp3 <= tmp4;
                            auto tmp6 = static_cast<float>(6.0);
                            auto tmp7 = tmp3 >= tmp6;
                            auto tmp8 = decltype(tmp5)(tmp5 | tmp7);
                            auto tmp10 = tmp8 ? tmp4 : tmp9;
                            auto tmp11 = decltype(tmp10)(tmp10 * tmp0);
                            tmp_acc0 = tmp_acc0 + tmp11;
                        }
                        out_ptr0[static_cast<long>(x1 + (702L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5616L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_threshold_backward_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (58L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (58L*x1)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (58L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp9 = tmp5 * tmp8;
                    tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    tmp_acc1_vec = tmp_acc1_vec + tmp9;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(56L); x0<static_cast<long>(58L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (58L*x1))];
                    auto tmp3 = in_ptr1[static_cast<long>(x0 + (58L*x1))];
                    auto tmp5 = in_ptr2[static_cast<long>(x0 + (58L*x1))];
                    auto tmp6 = in_ptr3[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    auto tmp4 = tmp2 ? tmp1 : tmp3;
                    auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                    auto tmp8 = decltype(tmp4)(tmp4 * tmp7);
                    tmp_acc0 = tmp_acc0 + tmp4;
                    tmp_acc1 = tmp_acc1 + tmp8;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(8.0);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 / tmp3;
            auto tmp5 = static_cast<float>(1e-05);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 + tmp6;
            auto tmp8 = tmp7.rsqrt();
            auto tmp9 = tmp0 * tmp8;
            tmp9.store(out_ptr2 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(56L); x0<static_cast<long>(58L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr4[static_cast<long>(x0)];
            auto tmp2 = static_cast<float>(8.0);
            auto tmp3 = tmp1 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
            auto tmp6 = 1 / std::sqrt(tmp5);
            auto tmp7 = decltype(tmp0)(tmp0 * tmp6);
            out_ptr2[static_cast<long>(x0)] = tmp7;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (58L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (58L*x0)));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (58L*x0)));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp25 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                auto tmp8 = tmp6 - tmp7;
                auto tmp10 = static_cast<float>(0.125);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp14 = static_cast<float>(8.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 / tmp15;
                auto tmp17 = static_cast<float>(1e-05);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 + tmp18;
                auto tmp20 = tmp19.rsqrt();
                auto tmp21 = tmp20 * tmp20;
                auto tmp22 = tmp12 * tmp21;
                auto tmp23 = tmp8 * tmp22;
                auto tmp24 = tmp5 - tmp23;
                auto tmp26 = tmp25 * tmp11;
                auto tmp27 = tmp24 - tmp26;
                auto tmp29 = tmp20 * tmp28;
                auto tmp30 = tmp27 * tmp29;
                tmp30.store(in_out_ptr0 + static_cast<long>(x1 + (58L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(56L); x1<static_cast<long>(58L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (58L*x0))];
                auto tmp3 = in_out_ptr0[static_cast<long>(x1 + (58L*x0))];
                auto tmp5 = in_ptr2[static_cast<long>(x1 + (58L*x0))];
                auto tmp6 = in_ptr3[static_cast<long>(x1)];
                auto tmp8 = out_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr4[static_cast<long>(x1)];
                auto tmp21 = out_ptr0[static_cast<long>(x1)];
                auto tmp24 = in_ptr5[static_cast<long>(x1)];
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp4 = tmp2 ? tmp1 : tmp3;
                auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                auto tmp9 = static_cast<float>(0.125);
                auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                auto tmp12 = static_cast<float>(8.0);
                auto tmp13 = tmp11 / tmp12;
                auto tmp14 = static_cast<float>(1e-05);
                auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                auto tmp16 = 1 / std::sqrt(tmp15);
                auto tmp17 = decltype(tmp16)(tmp16 * tmp16);
                auto tmp18 = decltype(tmp10)(tmp10 * tmp17);
                auto tmp19 = decltype(tmp7)(tmp7 * tmp18);
                auto tmp20 = decltype(tmp4)(tmp4 - tmp19);
                auto tmp22 = decltype(tmp21)(tmp21 * tmp9);
                auto tmp23 = decltype(tmp20)(tmp20 - tmp22);
                auto tmp25 = decltype(tmp16)(tmp16 * tmp24);
                auto tmp26 = decltype(tmp23)(tmp23 * tmp25);
                in_out_ptr0[static_cast<long>(x1 + (58L*x0))] = tmp26;
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0)
{
    auto in_ptr2 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(702L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x0 + (702L*x2) + (137592L*x1))];
                            auto tmp1 = in_ptr1[static_cast<long>(x0 + (702L*x1))];
                            auto tmp9 = in_ptr2[static_cast<long>(x0 + (702L*x2) + (137592L*x1))];
                            auto tmp12 = in_ptr3[static_cast<long>(x0 + (702L*x1))];
                            auto tmp16 = in_ptr4[static_cast<long>(x0 + (702L*x2) + (137592L*x1))];
                            auto tmp17 = in_ptr5[static_cast<long>(x0)];
                            auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = tmp3 <= tmp4;
                            auto tmp6 = static_cast<float>(6.0);
                            auto tmp7 = tmp3 >= tmp6;
                            auto tmp8 = decltype(tmp5)(tmp5 | tmp7);
                            auto tmp10 = tmp8 ? tmp4 : tmp9;
                            auto tmp11 = decltype(tmp10)(tmp10 * tmp2);
                            auto tmp13 = static_cast<float>(196.0);
                            auto tmp14 = tmp12 / tmp13;
                            auto tmp15 = decltype(tmp11)(tmp11 + tmp14);
                            auto tmp18 = decltype(tmp16)(tmp16 - tmp17);
                            auto tmp19 = decltype(tmp15)(tmp15 * tmp18);
                            tmp_acc0 = tmp_acc0 + tmp15;
                            tmp_acc1 = tmp_acc1 + tmp19;
                        }
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(702L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (702L*x1) + (137592L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (702L*x0))];
                        auto tmp9 = in_out_ptr0[static_cast<long>(x2 + (702L*x1) + (137592L*x0))];
                        auto tmp12 = in_ptr3[static_cast<long>(x2 + (702L*x0))];
                        auto tmp16 = in_ptr4[static_cast<long>(x2 + (702L*x1) + (137592L*x0))];
                        auto tmp17 = in_ptr5[static_cast<long>(x2)];
                        auto tmp19 = out_ptr1[static_cast<long>(x2)];
                        auto tmp22 = in_ptr6[static_cast<long>(x2)];
                        auto tmp27 = out_ptr0[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                        auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = tmp3 <= tmp4;
                        auto tmp6 = static_cast<float>(6.0);
                        auto tmp7 = tmp3 >= tmp6;
                        auto tmp8 = decltype(tmp5)(tmp5 | tmp7);
                        auto tmp10 = tmp8 ? tmp4 : tmp9;
                        auto tmp11 = decltype(tmp10)(tmp10 * tmp2);
                        auto tmp13 = static_cast<float>(196.0);
                        auto tmp14 = tmp12 / tmp13;
                        auto tmp15 = decltype(tmp11)(tmp11 + tmp14);
                        auto tmp18 = decltype(tmp16)(tmp16 - tmp17);
                        auto tmp20 = static_cast<float>(0.0006377551020408163);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp23 = decltype(tmp22)(tmp22 * tmp22);
                        auto tmp24 = decltype(tmp21)(tmp21 * tmp23);
                        auto tmp25 = decltype(tmp18)(tmp18 * tmp24);
                        auto tmp26 = decltype(tmp15)(tmp15 - tmp25);
                        auto tmp28 = decltype(tmp27)(tmp27 * tmp20);
                        auto tmp29 = decltype(tmp26)(tmp26 - tmp28);
                        in_out_ptr0[static_cast<long>(x2 + (702L*x1) + (137592L*x0))] = tmp29;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(696L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(696L); x0<static_cast<long>(702L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    in_out_ptr1[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(696L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (702L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (702L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(696L); x1<static_cast<long>(702L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (702L*x0))];
                    auto tmp1 = in_ptr6[static_cast<long>(x1)];
                    auto tmp2 = in_ptr7[static_cast<long>(x1)];
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                    in_out_ptr0[static_cast<long>(x1 + (702L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(696L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (702L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (702L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (702L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(696L); x0<static_cast<long>(702L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (702L*x1))];
                        auto tmp1 = in_ptr1[static_cast<long>(x0 + (702L*x1))];
                        auto tmp3 = in_ptr2[static_cast<long>(x0 + (702L*x1))];
                        auto tmp4 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = decltype(tmp2)(tmp2 * tmp5);
                        tmp_acc0 = tmp_acc0 + tmp2;
                        tmp_acc1 = tmp_acc1 + tmp6;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(696L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(696L); x0<static_cast<long>(702L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    out_ptr2[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(696L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (702L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (702L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (702L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.0006377551020408163);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (702L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(696L); x1<static_cast<long>(702L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (702L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (702L*x0))];
                    auto tmp3 = in_ptr2[static_cast<long>(x1 + (702L*x0))];
                    auto tmp4 = in_ptr3[static_cast<long>(x1)];
                    auto tmp6 = out_ptr1[static_cast<long>(x1)];
                    auto tmp9 = in_ptr4[static_cast<long>(x1)];
                    auto tmp14 = out_ptr0[static_cast<long>(x1)];
                    auto tmp17 = in_ptr5[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                    auto tmp7 = static_cast<float>(0.0006377551020408163);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp10 = decltype(tmp9)(tmp9 * tmp9);
                    auto tmp11 = decltype(tmp8)(tmp8 * tmp10);
                    auto tmp12 = decltype(tmp5)(tmp5 * tmp11);
                    auto tmp13 = decltype(tmp2)(tmp2 - tmp12);
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp7);
                    auto tmp16 = decltype(tmp13)(tmp13 - tmp15);
                    auto tmp18 = decltype(tmp9)(tmp9 * tmp17);
                    auto tmp19 = decltype(tmp16)(tmp16 * tmp18);
                    in_out_ptr0[static_cast<long>(x1 + (702L*x0))] = tmp19;
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_32 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(117L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp31 = in_ptr2[static_cast<long>(x0 + (117L*x1))];
                        auto tmp32 = in_ptr3[static_cast<long>(x0)];
                        auto tmp0 = c10::convert<long>(x0);
                        auto tmp1 = static_cast<long>(106);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr0[static_cast<long>(x0 + (128L*x1))];
                            auto tmp5 = in_ptr1[static_cast<long>(x0 + (117L*x1))];
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = tmp2 ? tmp7 : tmp8;
                        auto tmp10 = tmp0 < tmp1;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr0[static_cast<long>(x0 + (128L*x1))];
                            auto tmp13 = in_ptr1[static_cast<long>(x0 + (117L*x1))];
                            auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                            return tmp14;
                        }
                        ;
                        auto tmp15 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp16 = tmp10 ? tmp15 : tmp8;
                        auto tmp17 = decltype(tmp9)(tmp9 + tmp16);
                        auto tmp18 = [&]
                        {
                            auto tmp19 = in_ptr0[static_cast<long>(x0 + (128L*x1))];
                            auto tmp20 = in_ptr1[static_cast<long>(x0 + (117L*x1))];
                            auto tmp21 = decltype(tmp19)(tmp19 + tmp20);
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp2 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                        auto tmp23 = tmp2 ? tmp22 : tmp8;
                        auto tmp24 = [&]
                        {
                            auto tmp25 = in_ptr0[static_cast<long>(x0 + (128L*x1))];
                            auto tmp26 = in_ptr1[static_cast<long>(x0 + (117L*x1))];
                            auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp10 ? tmp24() : static_cast<decltype(tmp24())>(0.0);
                        auto tmp29 = tmp10 ? tmp28 : tmp8;
                        auto tmp30 = decltype(tmp23)(tmp23 + tmp29);
                        auto tmp33 = decltype(tmp31)(tmp31 - tmp32);
                        auto tmp34 = decltype(tmp30)(tmp30 * tmp33);
                        tmp_acc0 = tmp_acc0 + tmp17;
                        tmp_acc1 = tmp_acc1 + tmp34;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(112L); x0<static_cast<long>(117L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    out_ptr2[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(117L); x2+=static_cast<long>(1L))
                    {
                        auto tmp18 = in_ptr2[static_cast<long>(x2 + (117L*x1) + (22932L*x0))];
                        auto tmp19 = in_ptr3[static_cast<long>(x2)];
                        auto tmp21 = out_ptr1[static_cast<long>(x2)];
                        auto tmp24 = in_ptr4[static_cast<long>(x2)];
                        auto tmp29 = out_ptr0[static_cast<long>(x2)];
                        auto tmp32 = in_ptr5[static_cast<long>(x2)];
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(106);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (25088L*x0))];
                            auto tmp5 = in_ptr1[static_cast<long>(x2 + (117L*x1) + (22932L*x0))];
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp8 = static_cast<float>(0.0);
                        auto tmp9 = tmp2 ? tmp7 : tmp8;
                        auto tmp10 = tmp0 < tmp1;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (25088L*x0))];
                            auto tmp13 = in_ptr1[static_cast<long>(x2 + (117L*x1) + (22932L*x0))];
                            auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                            return tmp14;
                        }
                        ;
                        auto tmp15 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp16 = tmp10 ? tmp15 : tmp8;
                        auto tmp17 = decltype(tmp9)(tmp9 + tmp16);
                        auto tmp20 = decltype(tmp18)(tmp18 - tmp19);
                        auto tmp22 = static_cast<float>(0.0006377551020408163);
                        auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                        auto tmp25 = decltype(tmp24)(tmp24 * tmp24);
                        auto tmp26 = decltype(tmp23)(tmp23 * tmp25);
                        auto tmp27 = decltype(tmp20)(tmp20 * tmp26);
                        auto tmp28 = decltype(tmp17)(tmp17 - tmp27);
                        auto tmp30 = decltype(tmp29)(tmp29 * tmp22);
                        auto tmp31 = decltype(tmp28)(tmp28 - tmp30);
                        auto tmp33 = decltype(tmp24)(tmp24 * tmp32);
                        auto tmp34 = decltype(tmp31)(tmp31 * tmp33);
                        out_ptr3[static_cast<long>(x1 + (196L*x2) + (22932L*x0))] = tmp34;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(636L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (636L*x2) + (124656L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (636L*x0))];
                            auto tmp9 = in_ptr2[static_cast<long>(x1 + (636L*x2) + (124656L*x0))];
                            auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = tmp3 <= tmp4;
                            auto tmp6 = static_cast<float>(6.0);
                            auto tmp7 = tmp3 >= tmp6;
                            auto tmp8 = decltype(tmp5)(tmp5 | tmp7);
                            auto tmp10 = tmp8 ? tmp4 : tmp9;
                            auto tmp11 = decltype(tmp10)(tmp10 * tmp0);
                            tmp_acc0 = tmp_acc0 + tmp11;
                        }
                        out_ptr0[static_cast<long>(x1 + (636L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5088L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_threshold_backward_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (53L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (53L*x1)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (53L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp9 = tmp5 * tmp8;
                    tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    tmp_acc1_vec = tmp_acc1_vec + tmp9;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(48L); x0<static_cast<long>(53L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (53L*x1))];
                    auto tmp3 = in_ptr1[static_cast<long>(x0 + (53L*x1))];
                    auto tmp5 = in_ptr2[static_cast<long>(x0 + (53L*x1))];
                    auto tmp6 = in_ptr3[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    auto tmp4 = tmp2 ? tmp1 : tmp3;
                    auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                    auto tmp8 = decltype(tmp4)(tmp4 * tmp7);
                    tmp_acc0 = tmp_acc0 + tmp4;
                    tmp_acc1 = tmp_acc1 + tmp8;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(8.0);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 / tmp3;
            auto tmp5 = static_cast<float>(1e-05);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 + tmp6;
            auto tmp8 = tmp7.rsqrt();
            auto tmp9 = tmp0 * tmp8;
            tmp9.store(out_ptr2 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(48L); x0<static_cast<long>(53L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr4[static_cast<long>(x0)];
            auto tmp2 = static_cast<float>(8.0);
            auto tmp3 = tmp1 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
            auto tmp6 = 1 / std::sqrt(tmp5);
            auto tmp7 = decltype(tmp0)(tmp0 * tmp6);
            out_ptr2[static_cast<long>(x0)] = tmp7;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (53L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (53L*x0)));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (53L*x0)));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp25 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                auto tmp8 = tmp6 - tmp7;
                auto tmp10 = static_cast<float>(0.125);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp14 = static_cast<float>(8.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 / tmp15;
                auto tmp17 = static_cast<float>(1e-05);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 + tmp18;
                auto tmp20 = tmp19.rsqrt();
                auto tmp21 = tmp20 * tmp20;
                auto tmp22 = tmp12 * tmp21;
                auto tmp23 = tmp8 * tmp22;
                auto tmp24 = tmp5 - tmp23;
                auto tmp26 = tmp25 * tmp11;
                auto tmp27 = tmp24 - tmp26;
                auto tmp29 = tmp20 * tmp28;
                auto tmp30 = tmp27 * tmp29;
                tmp30.store(in_out_ptr0 + static_cast<long>(x1 + (53L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(48L); x1<static_cast<long>(53L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (53L*x0))];
                auto tmp3 = in_out_ptr0[static_cast<long>(x1 + (53L*x0))];
                auto tmp5 = in_ptr2[static_cast<long>(x1 + (53L*x0))];
                auto tmp6 = in_ptr3[static_cast<long>(x1)];
                auto tmp8 = out_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr4[static_cast<long>(x1)];
                auto tmp21 = out_ptr0[static_cast<long>(x1)];
                auto tmp24 = in_ptr5[static_cast<long>(x1)];
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp4 = tmp2 ? tmp1 : tmp3;
                auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                auto tmp9 = static_cast<float>(0.125);
                auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                auto tmp12 = static_cast<float>(8.0);
                auto tmp13 = tmp11 / tmp12;
                auto tmp14 = static_cast<float>(1e-05);
                auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                auto tmp16 = 1 / std::sqrt(tmp15);
                auto tmp17 = decltype(tmp16)(tmp16 * tmp16);
                auto tmp18 = decltype(tmp10)(tmp10 * tmp17);
                auto tmp19 = decltype(tmp7)(tmp7 * tmp18);
                auto tmp20 = decltype(tmp4)(tmp4 - tmp19);
                auto tmp22 = decltype(tmp21)(tmp21 * tmp9);
                auto tmp23 = decltype(tmp20)(tmp20 - tmp22);
                auto tmp25 = decltype(tmp16)(tmp16 * tmp24);
                auto tmp26 = decltype(tmp23)(tmp23 * tmp25);
                in_out_ptr0[static_cast<long>(x1 + (53L*x0))] = tmp26;
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0)
{
    auto in_ptr2 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(636L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x0 + (636L*x2) + (124656L*x1))];
                            auto tmp1 = in_ptr1[static_cast<long>(x0 + (636L*x1))];
                            auto tmp9 = in_ptr2[static_cast<long>(x0 + (636L*x2) + (124656L*x1))];
                            auto tmp12 = in_ptr3[static_cast<long>(x0 + (636L*x1))];
                            auto tmp16 = in_ptr4[static_cast<long>(x0 + (636L*x2) + (124656L*x1))];
                            auto tmp17 = in_ptr5[static_cast<long>(x0)];
                            auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = tmp3 <= tmp4;
                            auto tmp6 = static_cast<float>(6.0);
                            auto tmp7 = tmp3 >= tmp6;
                            auto tmp8 = decltype(tmp5)(tmp5 | tmp7);
                            auto tmp10 = tmp8 ? tmp4 : tmp9;
                            auto tmp11 = decltype(tmp10)(tmp10 * tmp2);
                            auto tmp13 = static_cast<float>(196.0);
                            auto tmp14 = tmp12 / tmp13;
                            auto tmp15 = decltype(tmp11)(tmp11 + tmp14);
                            auto tmp18 = decltype(tmp16)(tmp16 - tmp17);
                            auto tmp19 = decltype(tmp15)(tmp15 * tmp18);
                            tmp_acc0 = tmp_acc0 + tmp15;
                            tmp_acc1 = tmp_acc1 + tmp19;
                        }
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(636L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (636L*x1) + (124656L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (636L*x0))];
                        auto tmp9 = in_out_ptr0[static_cast<long>(x2 + (636L*x1) + (124656L*x0))];
                        auto tmp12 = in_ptr3[static_cast<long>(x2 + (636L*x0))];
                        auto tmp16 = in_ptr4[static_cast<long>(x2 + (636L*x1) + (124656L*x0))];
                        auto tmp17 = in_ptr5[static_cast<long>(x2)];
                        auto tmp19 = out_ptr1[static_cast<long>(x2)];
                        auto tmp22 = in_ptr6[static_cast<long>(x2)];
                        auto tmp27 = out_ptr0[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                        auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = tmp3 <= tmp4;
                        auto tmp6 = static_cast<float>(6.0);
                        auto tmp7 = tmp3 >= tmp6;
                        auto tmp8 = decltype(tmp5)(tmp5 | tmp7);
                        auto tmp10 = tmp8 ? tmp4 : tmp9;
                        auto tmp11 = decltype(tmp10)(tmp10 * tmp2);
                        auto tmp13 = static_cast<float>(196.0);
                        auto tmp14 = tmp12 / tmp13;
                        auto tmp15 = decltype(tmp11)(tmp11 + tmp14);
                        auto tmp18 = decltype(tmp16)(tmp16 - tmp17);
                        auto tmp20 = static_cast<float>(0.0006377551020408163);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp23 = decltype(tmp22)(tmp22 * tmp22);
                        auto tmp24 = decltype(tmp21)(tmp21 * tmp23);
                        auto tmp25 = decltype(tmp18)(tmp18 * tmp24);
                        auto tmp26 = decltype(tmp15)(tmp15 - tmp25);
                        auto tmp28 = decltype(tmp27)(tmp27 * tmp20);
                        auto tmp29 = decltype(tmp26)(tmp26 - tmp28);
                        in_out_ptr0[static_cast<long>(x2 + (636L*x1) + (124656L*x0))] = tmp29;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(632L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(632L); x0<static_cast<long>(636L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    in_out_ptr1[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(632L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (636L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (636L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(632L); x1<static_cast<long>(636L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (636L*x0))];
                    auto tmp1 = in_ptr6[static_cast<long>(x1)];
                    auto tmp2 = in_ptr7[static_cast<long>(x1)];
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                    in_out_ptr0[static_cast<long>(x1 + (636L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(632L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (636L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (636L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (636L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(632L); x0<static_cast<long>(636L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (636L*x1))];
                        auto tmp1 = in_ptr1[static_cast<long>(x0 + (636L*x1))];
                        auto tmp3 = in_ptr2[static_cast<long>(x0 + (636L*x1))];
                        auto tmp4 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = decltype(tmp2)(tmp2 * tmp5);
                        tmp_acc0 = tmp_acc0 + tmp2;
                        tmp_acc1 = tmp_acc1 + tmp6;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(632L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(632L); x0<static_cast<long>(636L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    out_ptr2[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(632L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (636L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (636L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (636L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.0006377551020408163);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (636L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(632L); x1<static_cast<long>(636L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (636L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (636L*x0))];
                    auto tmp3 = in_ptr2[static_cast<long>(x1 + (636L*x0))];
                    auto tmp4 = in_ptr3[static_cast<long>(x1)];
                    auto tmp6 = out_ptr1[static_cast<long>(x1)];
                    auto tmp9 = in_ptr4[static_cast<long>(x1)];
                    auto tmp14 = out_ptr0[static_cast<long>(x1)];
                    auto tmp17 = in_ptr5[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                    auto tmp7 = static_cast<float>(0.0006377551020408163);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp10 = decltype(tmp9)(tmp9 * tmp9);
                    auto tmp11 = decltype(tmp8)(tmp8 * tmp10);
                    auto tmp12 = decltype(tmp5)(tmp5 * tmp11);
                    auto tmp13 = decltype(tmp2)(tmp2 - tmp12);
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp7);
                    auto tmp16 = decltype(tmp13)(tmp13 - tmp15);
                    auto tmp18 = decltype(tmp9)(tmp9 * tmp17);
                    auto tmp19 = decltype(tmp16)(tmp16 * tmp18);
                    in_out_ptr0[static_cast<long>(x1 + (636L*x0))] = tmp19;
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_slice_backward_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(106L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp39 = in_ptr3[static_cast<long>(x0 + (106L*x1))];
                        auto tmp40 = in_ptr4[static_cast<long>(x0)];
                        auto tmp0 = c10::convert<long>(x0);
                        auto tmp1 = static_cast<long>(95);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr0[static_cast<long>(x0 + (128L*x1))];
                            auto tmp5 = in_ptr1[static_cast<long>(x0 + (117L*x1))];
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            auto tmp7 = in_ptr2[static_cast<long>(x0 + (106L*x1))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp10 = static_cast<float>(0.0);
                        auto tmp11 = tmp2 ? tmp9 : tmp10;
                        auto tmp12 = tmp0 < tmp1;
                        auto tmp13 = [&]
                        {
                            auto tmp14 = in_ptr0[static_cast<long>(x0 + (128L*x1))];
                            auto tmp15 = in_ptr1[static_cast<long>(x0 + (117L*x1))];
                            auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                            auto tmp17 = in_ptr2[static_cast<long>(x0 + (106L*x1))];
                            auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                            return tmp18;
                        }
                        ;
                        auto tmp19 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                        auto tmp20 = tmp12 ? tmp19 : tmp10;
                        auto tmp21 = decltype(tmp11)(tmp11 + tmp20);
                        auto tmp22 = [&]
                        {
                            auto tmp23 = in_ptr0[static_cast<long>(x0 + (128L*x1))];
                            auto tmp24 = in_ptr1[static_cast<long>(x0 + (117L*x1))];
                            auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                            auto tmp26 = in_ptr2[static_cast<long>(x0 + (106L*x1))];
                            auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp2 ? tmp22() : static_cast<decltype(tmp22())>(0.0);
                        auto tmp29 = tmp2 ? tmp28 : tmp10;
                        auto tmp30 = [&]
                        {
                            auto tmp31 = in_ptr0[static_cast<long>(x0 + (128L*x1))];
                            auto tmp32 = in_ptr1[static_cast<long>(x0 + (117L*x1))];
                            auto tmp33 = decltype(tmp31)(tmp31 + tmp32);
                            auto tmp34 = in_ptr2[static_cast<long>(x0 + (106L*x1))];
                            auto tmp35 = decltype(tmp33)(tmp33 + tmp34);
                            return tmp35;
                        }
                        ;
                        auto tmp36 = tmp12 ? tmp30() : static_cast<decltype(tmp30())>(0.0);
                        auto tmp37 = tmp12 ? tmp36 : tmp10;
                        auto tmp38 = decltype(tmp29)(tmp29 + tmp37);
                        auto tmp41 = decltype(tmp39)(tmp39 - tmp40);
                        auto tmp42 = decltype(tmp38)(tmp38 * tmp41);
                        tmp_acc0 = tmp_acc0 + tmp21;
                        tmp_acc1 = tmp_acc1 + tmp42;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(106L); x2+=static_cast<long>(1L))
                    {
                        auto tmp22 = in_ptr3[static_cast<long>(x2 + (106L*x1) + (20776L*x0))];
                        auto tmp23 = in_ptr4[static_cast<long>(x2)];
                        auto tmp25 = out_ptr1[static_cast<long>(x2)];
                        auto tmp28 = in_ptr5[static_cast<long>(x2)];
                        auto tmp33 = out_ptr0[static_cast<long>(x2)];
                        auto tmp36 = in_ptr6[static_cast<long>(x2)];
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(95);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (25088L*x0))];
                            auto tmp5 = in_ptr1[static_cast<long>(x2 + (117L*x1) + (22932L*x0))];
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            auto tmp7 = in_ptr2[static_cast<long>(x2 + (106L*x1) + (20776L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            return tmp8;
                        }
                        ;
                        auto tmp9 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp10 = static_cast<float>(0.0);
                        auto tmp11 = tmp2 ? tmp9 : tmp10;
                        auto tmp12 = tmp0 < tmp1;
                        auto tmp13 = [&]
                        {
                            auto tmp14 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (25088L*x0))];
                            auto tmp15 = in_ptr1[static_cast<long>(x2 + (117L*x1) + (22932L*x0))];
                            auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                            auto tmp17 = in_ptr2[static_cast<long>(x2 + (106L*x1) + (20776L*x0))];
                            auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                            return tmp18;
                        }
                        ;
                        auto tmp19 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                        auto tmp20 = tmp12 ? tmp19 : tmp10;
                        auto tmp21 = decltype(tmp11)(tmp11 + tmp20);
                        auto tmp24 = decltype(tmp22)(tmp22 - tmp23);
                        auto tmp26 = static_cast<float>(0.0006377551020408163);
                        auto tmp27 = decltype(tmp25)(tmp25 * tmp26);
                        auto tmp29 = decltype(tmp28)(tmp28 * tmp28);
                        auto tmp30 = decltype(tmp27)(tmp27 * tmp29);
                        auto tmp31 = decltype(tmp24)(tmp24 * tmp30);
                        auto tmp32 = decltype(tmp21)(tmp21 - tmp31);
                        auto tmp34 = decltype(tmp33)(tmp33 * tmp26);
                        auto tmp35 = decltype(tmp32)(tmp32 - tmp34);
                        auto tmp37 = decltype(tmp28)(tmp28 * tmp36);
                        auto tmp38 = decltype(tmp35)(tmp35 * tmp37);
                        out_ptr2[static_cast<long>(x1 + (196L*x2) + (20776L*x0))] = tmp38;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(104L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(104L); x0<static_cast<long>(106L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    in_out_ptr0[static_cast<long>(x0)] = tmp2;
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(570L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (570L*x2) + (111720L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (570L*x0))];
                            auto tmp9 = in_ptr2[static_cast<long>(x1 + (570L*x2) + (111720L*x0))];
                            auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = tmp3 <= tmp4;
                            auto tmp6 = static_cast<float>(6.0);
                            auto tmp7 = tmp3 >= tmp6;
                            auto tmp8 = decltype(tmp5)(tmp5 | tmp7);
                            auto tmp10 = tmp8 ? tmp4 : tmp9;
                            auto tmp11 = decltype(tmp10)(tmp10 * tmp0);
                            tmp_acc0 = tmp_acc0 + tmp11;
                        }
                        out_ptr0[static_cast<long>(x1 + (570L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4560L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_threshold_backward_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (47L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (47L*x1)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (47L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp9 = tmp5 * tmp8;
                    tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    tmp_acc1_vec = tmp_acc1_vec + tmp9;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(40L); x0<static_cast<long>(47L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (47L*x1))];
                    auto tmp3 = in_ptr1[static_cast<long>(x0 + (47L*x1))];
                    auto tmp5 = in_ptr2[static_cast<long>(x0 + (47L*x1))];
                    auto tmp6 = in_ptr3[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    auto tmp4 = tmp2 ? tmp1 : tmp3;
                    auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                    auto tmp8 = decltype(tmp4)(tmp4 * tmp7);
                    tmp_acc0 = tmp_acc0 + tmp4;
                    tmp_acc1 = tmp_acc1 + tmp8;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(8.0);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 / tmp3;
            auto tmp5 = static_cast<float>(1e-05);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 + tmp6;
            auto tmp8 = tmp7.rsqrt();
            auto tmp9 = tmp0 * tmp8;
            tmp9.store(out_ptr2 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(40L); x0<static_cast<long>(47L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr4[static_cast<long>(x0)];
            auto tmp2 = static_cast<float>(8.0);
            auto tmp3 = tmp1 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
            auto tmp6 = 1 / std::sqrt(tmp5);
            auto tmp7 = decltype(tmp0)(tmp0 * tmp6);
            out_ptr2[static_cast<long>(x0)] = tmp7;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (47L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (47L*x0)));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (47L*x0)));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp25 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                auto tmp8 = tmp6 - tmp7;
                auto tmp10 = static_cast<float>(0.125);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp14 = static_cast<float>(8.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 / tmp15;
                auto tmp17 = static_cast<float>(1e-05);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 + tmp18;
                auto tmp20 = tmp19.rsqrt();
                auto tmp21 = tmp20 * tmp20;
                auto tmp22 = tmp12 * tmp21;
                auto tmp23 = tmp8 * tmp22;
                auto tmp24 = tmp5 - tmp23;
                auto tmp26 = tmp25 * tmp11;
                auto tmp27 = tmp24 - tmp26;
                auto tmp29 = tmp20 * tmp28;
                auto tmp30 = tmp27 * tmp29;
                tmp30.store(in_out_ptr0 + static_cast<long>(x1 + (47L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(40L); x1<static_cast<long>(47L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (47L*x0))];
                auto tmp3 = in_out_ptr0[static_cast<long>(x1 + (47L*x0))];
                auto tmp5 = in_ptr2[static_cast<long>(x1 + (47L*x0))];
                auto tmp6 = in_ptr3[static_cast<long>(x1)];
                auto tmp8 = out_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr4[static_cast<long>(x1)];
                auto tmp21 = out_ptr0[static_cast<long>(x1)];
                auto tmp24 = in_ptr5[static_cast<long>(x1)];
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp4 = tmp2 ? tmp1 : tmp3;
                auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                auto tmp9 = static_cast<float>(0.125);
                auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                auto tmp12 = static_cast<float>(8.0);
                auto tmp13 = tmp11 / tmp12;
                auto tmp14 = static_cast<float>(1e-05);
                auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                auto tmp16 = 1 / std::sqrt(tmp15);
                auto tmp17 = decltype(tmp16)(tmp16 * tmp16);
                auto tmp18 = decltype(tmp10)(tmp10 * tmp17);
                auto tmp19 = decltype(tmp7)(tmp7 * tmp18);
                auto tmp20 = decltype(tmp4)(tmp4 - tmp19);
                auto tmp22 = decltype(tmp21)(tmp21 * tmp9);
                auto tmp23 = decltype(tmp20)(tmp20 - tmp22);
                auto tmp25 = decltype(tmp16)(tmp16 * tmp24);
                auto tmp26 = decltype(tmp23)(tmp23 * tmp25);
                in_out_ptr0[static_cast<long>(x1 + (47L*x0))] = tmp26;
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0)
{
    auto in_ptr2 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(570L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x0 + (570L*x2) + (111720L*x1))];
                            auto tmp1 = in_ptr1[static_cast<long>(x0 + (570L*x1))];
                            auto tmp9 = in_ptr2[static_cast<long>(x0 + (570L*x2) + (111720L*x1))];
                            auto tmp12 = in_ptr3[static_cast<long>(x0 + (570L*x1))];
                            auto tmp16 = in_ptr4[static_cast<long>(x0 + (570L*x2) + (111720L*x1))];
                            auto tmp17 = in_ptr5[static_cast<long>(x0)];
                            auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = tmp3 <= tmp4;
                            auto tmp6 = static_cast<float>(6.0);
                            auto tmp7 = tmp3 >= tmp6;
                            auto tmp8 = decltype(tmp5)(tmp5 | tmp7);
                            auto tmp10 = tmp8 ? tmp4 : tmp9;
                            auto tmp11 = decltype(tmp10)(tmp10 * tmp2);
                            auto tmp13 = static_cast<float>(196.0);
                            auto tmp14 = tmp12 / tmp13;
                            auto tmp15 = decltype(tmp11)(tmp11 + tmp14);
                            auto tmp18 = decltype(tmp16)(tmp16 - tmp17);
                            auto tmp19 = decltype(tmp15)(tmp15 * tmp18);
                            tmp_acc0 = tmp_acc0 + tmp15;
                            tmp_acc1 = tmp_acc1 + tmp19;
                        }
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(570L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (570L*x1) + (111720L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (570L*x0))];
                        auto tmp9 = in_out_ptr0[static_cast<long>(x2 + (570L*x1) + (111720L*x0))];
                        auto tmp12 = in_ptr3[static_cast<long>(x2 + (570L*x0))];
                        auto tmp16 = in_ptr4[static_cast<long>(x2 + (570L*x1) + (111720L*x0))];
                        auto tmp17 = in_ptr5[static_cast<long>(x2)];
                        auto tmp19 = out_ptr1[static_cast<long>(x2)];
                        auto tmp22 = in_ptr6[static_cast<long>(x2)];
                        auto tmp27 = out_ptr0[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                        auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = tmp3 <= tmp4;
                        auto tmp6 = static_cast<float>(6.0);
                        auto tmp7 = tmp3 >= tmp6;
                        auto tmp8 = decltype(tmp5)(tmp5 | tmp7);
                        auto tmp10 = tmp8 ? tmp4 : tmp9;
                        auto tmp11 = decltype(tmp10)(tmp10 * tmp2);
                        auto tmp13 = static_cast<float>(196.0);
                        auto tmp14 = tmp12 / tmp13;
                        auto tmp15 = decltype(tmp11)(tmp11 + tmp14);
                        auto tmp18 = decltype(tmp16)(tmp16 - tmp17);
                        auto tmp20 = static_cast<float>(0.0006377551020408163);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp23 = decltype(tmp22)(tmp22 * tmp22);
                        auto tmp24 = decltype(tmp21)(tmp21 * tmp23);
                        auto tmp25 = decltype(tmp18)(tmp18 * tmp24);
                        auto tmp26 = decltype(tmp15)(tmp15 - tmp25);
                        auto tmp28 = decltype(tmp27)(tmp27 * tmp20);
                        auto tmp29 = decltype(tmp26)(tmp26 - tmp28);
                        in_out_ptr0[static_cast<long>(x2 + (570L*x1) + (111720L*x0))] = tmp29;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(568L); x0<static_cast<long>(570L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    in_out_ptr1[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(568L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (570L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (570L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(568L); x1<static_cast<long>(570L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (570L*x0))];
                    auto tmp1 = in_ptr6[static_cast<long>(x1)];
                    auto tmp2 = in_ptr7[static_cast<long>(x1)];
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                    in_out_ptr0[static_cast<long>(x1 + (570L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(568L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (570L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (570L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (570L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(568L); x0<static_cast<long>(570L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (570L*x1))];
                        auto tmp1 = in_ptr1[static_cast<long>(x0 + (570L*x1))];
                        auto tmp3 = in_ptr2[static_cast<long>(x0 + (570L*x1))];
                        auto tmp4 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = decltype(tmp2)(tmp2 * tmp5);
                        tmp_acc0 = tmp_acc0 + tmp2;
                        tmp_acc1 = tmp_acc1 + tmp6;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(568L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(568L); x0<static_cast<long>(570L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    out_ptr2[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(568L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (570L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (570L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (570L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.0006377551020408163);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (570L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(568L); x1<static_cast<long>(570L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (570L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (570L*x0))];
                    auto tmp3 = in_ptr2[static_cast<long>(x1 + (570L*x0))];
                    auto tmp4 = in_ptr3[static_cast<long>(x1)];
                    auto tmp6 = out_ptr1[static_cast<long>(x1)];
                    auto tmp9 = in_ptr4[static_cast<long>(x1)];
                    auto tmp14 = out_ptr0[static_cast<long>(x1)];
                    auto tmp17 = in_ptr5[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                    auto tmp7 = static_cast<float>(0.0006377551020408163);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp10 = decltype(tmp9)(tmp9 * tmp9);
                    auto tmp11 = decltype(tmp8)(tmp8 * tmp10);
                    auto tmp12 = decltype(tmp5)(tmp5 * tmp11);
                    auto tmp13 = decltype(tmp2)(tmp2 - tmp12);
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp7);
                    auto tmp16 = decltype(tmp13)(tmp13 - tmp15);
                    auto tmp18 = decltype(tmp9)(tmp9 * tmp17);
                    auto tmp19 = decltype(tmp16)(tmp16 * tmp18);
                    in_out_ptr0[static_cast<long>(x1 + (570L*x0))] = tmp19;
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(95L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp47 = in_ptr4[static_cast<long>(x0 + (95L*x1))];
                        auto tmp48 = in_ptr5[static_cast<long>(x0)];
                        auto tmp0 = c10::convert<long>(x0);
                        auto tmp1 = static_cast<long>(84);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr0[static_cast<long>(x0 + (128L*x1))];
                            auto tmp5 = in_ptr1[static_cast<long>(x0 + (117L*x1))];
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            auto tmp7 = in_ptr2[static_cast<long>(x0 + (106L*x1))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp9 = in_ptr3[static_cast<long>(x0 + (95L*x1))];
                            auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                            return tmp10;
                        }
                        ;
                        auto tmp11 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp12 = static_cast<float>(0.0);
                        auto tmp13 = tmp2 ? tmp11 : tmp12;
                        auto tmp14 = tmp0 < tmp1;
                        auto tmp15 = [&]
                        {
                            auto tmp16 = in_ptr0[static_cast<long>(x0 + (128L*x1))];
                            auto tmp17 = in_ptr1[static_cast<long>(x0 + (117L*x1))];
                            auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                            auto tmp19 = in_ptr2[static_cast<long>(x0 + (106L*x1))];
                            auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                            auto tmp21 = in_ptr3[static_cast<long>(x0 + (95L*x1))];
                            auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                            return tmp22;
                        }
                        ;
                        auto tmp23 = tmp14 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp24 = tmp14 ? tmp23 : tmp12;
                        auto tmp25 = decltype(tmp13)(tmp13 + tmp24);
                        auto tmp26 = [&]
                        {
                            auto tmp27 = in_ptr0[static_cast<long>(x0 + (128L*x1))];
                            auto tmp28 = in_ptr1[static_cast<long>(x0 + (117L*x1))];
                            auto tmp29 = decltype(tmp27)(tmp27 + tmp28);
                            auto tmp30 = in_ptr2[static_cast<long>(x0 + (106L*x1))];
                            auto tmp31 = decltype(tmp29)(tmp29 + tmp30);
                            auto tmp32 = in_ptr3[static_cast<long>(x0 + (95L*x1))];
                            auto tmp33 = decltype(tmp31)(tmp31 + tmp32);
                            return tmp33;
                        }
                        ;
                        auto tmp34 = tmp2 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                        auto tmp35 = tmp2 ? tmp34 : tmp12;
                        auto tmp36 = [&]
                        {
                            auto tmp37 = in_ptr0[static_cast<long>(x0 + (128L*x1))];
                            auto tmp38 = in_ptr1[static_cast<long>(x0 + (117L*x1))];
                            auto tmp39 = decltype(tmp37)(tmp37 + tmp38);
                            auto tmp40 = in_ptr2[static_cast<long>(x0 + (106L*x1))];
                            auto tmp41 = decltype(tmp39)(tmp39 + tmp40);
                            auto tmp42 = in_ptr3[static_cast<long>(x0 + (95L*x1))];
                            auto tmp43 = decltype(tmp41)(tmp41 + tmp42);
                            return tmp43;
                        }
                        ;
                        auto tmp44 = tmp14 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                        auto tmp45 = tmp14 ? tmp44 : tmp12;
                        auto tmp46 = decltype(tmp35)(tmp35 + tmp45);
                        auto tmp49 = decltype(tmp47)(tmp47 - tmp48);
                        auto tmp50 = decltype(tmp46)(tmp46 * tmp49);
                        tmp_acc0 = tmp_acc0 + tmp25;
                        tmp_acc1 = tmp_acc1 + tmp50;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
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
                    auto tmp26 = in_ptr4[static_cast<long>(x1 + (95L*x0))];
                    auto tmp27 = in_ptr5[static_cast<long>(x1)];
                    auto tmp29 = out_ptr1[static_cast<long>(x1)];
                    auto tmp32 = in_ptr6[static_cast<long>(x1)];
                    auto tmp37 = out_ptr0[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(84);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp5 = in_ptr1[static_cast<long>(x1 + (117L*x0))];
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp7 = in_ptr2[static_cast<long>(x1 + (106L*x0))];
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = in_ptr3[static_cast<long>(x1 + (95L*x0))];
                        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                        return tmp10;
                    }
                    ;
                    auto tmp11 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                    auto tmp12 = static_cast<float>(0.0);
                    auto tmp13 = tmp2 ? tmp11 : tmp12;
                    auto tmp14 = tmp0 < tmp1;
                    auto tmp15 = [&]
                    {
                        auto tmp16 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp17 = in_ptr1[static_cast<long>(x1 + (117L*x0))];
                        auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                        auto tmp19 = in_ptr2[static_cast<long>(x1 + (106L*x0))];
                        auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                        auto tmp21 = in_ptr3[static_cast<long>(x1 + (95L*x0))];
                        auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                        return tmp22;
                    }
                    ;
                    auto tmp23 = tmp14 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                    auto tmp24 = tmp14 ? tmp23 : tmp12;
                    auto tmp25 = decltype(tmp13)(tmp13 + tmp24);
                    auto tmp28 = decltype(tmp26)(tmp26 - tmp27);
                    auto tmp30 = static_cast<float>(0.0006377551020408163);
                    auto tmp31 = decltype(tmp29)(tmp29 * tmp30);
                    auto tmp33 = decltype(tmp32)(tmp32 * tmp32);
                    auto tmp34 = decltype(tmp31)(tmp31 * tmp33);
                    auto tmp35 = decltype(tmp28)(tmp28 * tmp34);
                    auto tmp36 = decltype(tmp25)(tmp25 - tmp35);
                    auto tmp38 = decltype(tmp37)(tmp37 * tmp30);
                    auto tmp39 = decltype(tmp36)(tmp36 - tmp38);
                    out_ptr2[static_cast<long>(x1 + (95L*x0))] = tmp39;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(88L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(88L); x0<static_cast<long>(95L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    in_out_ptr0[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(88L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(x1 + (95L*x2) + (18620L*x0)), static_cast<long>(95L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = in_ptr7[static_cast<long>(x1 + x1_inner)];
                            auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp1 * tmp5;
                            tmp6.store(out_ptr3 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (18620L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (95L*x2) + (18620L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                        auto tmp3 = tmp1 * tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp4.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr3[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (18620L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(88L); x1<static_cast<long>(95L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr2[static_cast<long>(x1 + (95L*x2) + (18620L*x0))];
                        auto tmp1 = in_ptr6[static_cast<long>(x1)];
                        auto tmp2 = in_ptr7[static_cast<long>(x1)];
                        auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                        auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                        out_ptr3[static_cast<long>(x2 + (196L*x1) + (18620L*x0))] = tmp4;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(504L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (504L*x2) + (98784L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (504L*x0))];
                            auto tmp9 = in_ptr2[static_cast<long>(x1 + (504L*x2) + (98784L*x0))];
                            auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = tmp3 <= tmp4;
                            auto tmp6 = static_cast<float>(6.0);
                            auto tmp7 = tmp3 >= tmp6;
                            auto tmp8 = decltype(tmp5)(tmp5 | tmp7);
                            auto tmp10 = tmp8 ? tmp4 : tmp9;
                            auto tmp11 = decltype(tmp10)(tmp10 * tmp0);
                            tmp_acc0 = tmp_acc0 + tmp11;
                        }
                        out_ptr0[static_cast<long>(x1 + (504L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4032L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_threshold_backward_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (42L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (42L*x1)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (42L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp9 = tmp5 * tmp8;
                    tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    tmp_acc1_vec = tmp_acc1_vec + tmp9;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(40L); x0<static_cast<long>(42L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (42L*x1))];
                    auto tmp3 = in_ptr1[static_cast<long>(x0 + (42L*x1))];
                    auto tmp5 = in_ptr2[static_cast<long>(x0 + (42L*x1))];
                    auto tmp6 = in_ptr3[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    auto tmp4 = tmp2 ? tmp1 : tmp3;
                    auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                    auto tmp8 = decltype(tmp4)(tmp4 * tmp7);
                    tmp_acc0 = tmp_acc0 + tmp4;
                    tmp_acc1 = tmp_acc1 + tmp8;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(8.0);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 / tmp3;
            auto tmp5 = static_cast<float>(1e-05);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 + tmp6;
            auto tmp8 = tmp7.rsqrt();
            auto tmp9 = tmp0 * tmp8;
            tmp9.store(out_ptr2 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(40L); x0<static_cast<long>(42L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr4[static_cast<long>(x0)];
            auto tmp2 = static_cast<float>(8.0);
            auto tmp3 = tmp1 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
            auto tmp6 = 1 / std::sqrt(tmp5);
            auto tmp7 = decltype(tmp0)(tmp0 * tmp6);
            out_ptr2[static_cast<long>(x0)] = tmp7;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (42L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (42L*x0)));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (42L*x0)));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp25 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                auto tmp8 = tmp6 - tmp7;
                auto tmp10 = static_cast<float>(0.125);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp14 = static_cast<float>(8.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 / tmp15;
                auto tmp17 = static_cast<float>(1e-05);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 + tmp18;
                auto tmp20 = tmp19.rsqrt();
                auto tmp21 = tmp20 * tmp20;
                auto tmp22 = tmp12 * tmp21;
                auto tmp23 = tmp8 * tmp22;
                auto tmp24 = tmp5 - tmp23;
                auto tmp26 = tmp25 * tmp11;
                auto tmp27 = tmp24 - tmp26;
                auto tmp29 = tmp20 * tmp28;
                auto tmp30 = tmp27 * tmp29;
                tmp30.store(in_out_ptr0 + static_cast<long>(x1 + (42L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(40L); x1<static_cast<long>(42L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (42L*x0))];
                auto tmp3 = in_out_ptr0[static_cast<long>(x1 + (42L*x0))];
                auto tmp5 = in_ptr2[static_cast<long>(x1 + (42L*x0))];
                auto tmp6 = in_ptr3[static_cast<long>(x1)];
                auto tmp8 = out_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr4[static_cast<long>(x1)];
                auto tmp21 = out_ptr0[static_cast<long>(x1)];
                auto tmp24 = in_ptr5[static_cast<long>(x1)];
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp4 = tmp2 ? tmp1 : tmp3;
                auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                auto tmp9 = static_cast<float>(0.125);
                auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                auto tmp12 = static_cast<float>(8.0);
                auto tmp13 = tmp11 / tmp12;
                auto tmp14 = static_cast<float>(1e-05);
                auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                auto tmp16 = 1 / std::sqrt(tmp15);
                auto tmp17 = decltype(tmp16)(tmp16 * tmp16);
                auto tmp18 = decltype(tmp10)(tmp10 * tmp17);
                auto tmp19 = decltype(tmp7)(tmp7 * tmp18);
                auto tmp20 = decltype(tmp4)(tmp4 - tmp19);
                auto tmp22 = decltype(tmp21)(tmp21 * tmp9);
                auto tmp23 = decltype(tmp20)(tmp20 - tmp22);
                auto tmp25 = decltype(tmp16)(tmp16 * tmp24);
                auto tmp26 = decltype(tmp23)(tmp23 * tmp25);
                in_out_ptr0[static_cast<long>(x1 + (42L*x0))] = tmp26;
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0)
{
    auto in_ptr2 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(504L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x0 + (504L*x2) + (98784L*x1))];
                            auto tmp1 = in_ptr1[static_cast<long>(x0 + (504L*x1))];
                            auto tmp9 = in_ptr2[static_cast<long>(x0 + (504L*x2) + (98784L*x1))];
                            auto tmp12 = in_ptr3[static_cast<long>(x0 + (504L*x1))];
                            auto tmp16 = in_ptr4[static_cast<long>(x0 + (504L*x2) + (98784L*x1))];
                            auto tmp17 = in_ptr5[static_cast<long>(x0)];
                            auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = tmp3 <= tmp4;
                            auto tmp6 = static_cast<float>(6.0);
                            auto tmp7 = tmp3 >= tmp6;
                            auto tmp8 = decltype(tmp5)(tmp5 | tmp7);
                            auto tmp10 = tmp8 ? tmp4 : tmp9;
                            auto tmp11 = decltype(tmp10)(tmp10 * tmp2);
                            auto tmp13 = static_cast<float>(196.0);
                            auto tmp14 = tmp12 / tmp13;
                            auto tmp15 = decltype(tmp11)(tmp11 + tmp14);
                            auto tmp18 = decltype(tmp16)(tmp16 - tmp17);
                            auto tmp19 = decltype(tmp15)(tmp15 * tmp18);
                            tmp_acc0 = tmp_acc0 + tmp15;
                            tmp_acc1 = tmp_acc1 + tmp19;
                        }
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(504L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (504L*x1) + (98784L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (504L*x0))];
                        auto tmp9 = in_out_ptr0[static_cast<long>(x2 + (504L*x1) + (98784L*x0))];
                        auto tmp12 = in_ptr3[static_cast<long>(x2 + (504L*x0))];
                        auto tmp16 = in_ptr4[static_cast<long>(x2 + (504L*x1) + (98784L*x0))];
                        auto tmp17 = in_ptr5[static_cast<long>(x2)];
                        auto tmp19 = out_ptr1[static_cast<long>(x2)];
                        auto tmp22 = in_ptr6[static_cast<long>(x2)];
                        auto tmp27 = out_ptr0[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                        auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = tmp3 <= tmp4;
                        auto tmp6 = static_cast<float>(6.0);
                        auto tmp7 = tmp3 >= tmp6;
                        auto tmp8 = decltype(tmp5)(tmp5 | tmp7);
                        auto tmp10 = tmp8 ? tmp4 : tmp9;
                        auto tmp11 = decltype(tmp10)(tmp10 * tmp2);
                        auto tmp13 = static_cast<float>(196.0);
                        auto tmp14 = tmp12 / tmp13;
                        auto tmp15 = decltype(tmp11)(tmp11 + tmp14);
                        auto tmp18 = decltype(tmp16)(tmp16 - tmp17);
                        auto tmp20 = static_cast<float>(0.0006377551020408163);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp23 = decltype(tmp22)(tmp22 * tmp22);
                        auto tmp24 = decltype(tmp21)(tmp21 * tmp23);
                        auto tmp25 = decltype(tmp18)(tmp18 * tmp24);
                        auto tmp26 = decltype(tmp15)(tmp15 - tmp25);
                        auto tmp28 = decltype(tmp27)(tmp27 * tmp20);
                        auto tmp29 = decltype(tmp26)(tmp26 - tmp28);
                        in_out_ptr0[static_cast<long>(x2 + (504L*x1) + (98784L*x0))] = tmp29;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(504L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(504L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (504L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (504L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(504L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (504L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (504L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (504L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(504L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(504L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (504L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (504L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (504L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.0006377551020408163);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (504L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
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
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (117L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (106L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (95L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (84L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 + tmp7;
                    tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (84L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(80L); x1<static_cast<long>(84L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (117L*x0))];
                    auto tmp3 = in_ptr2[static_cast<long>(x1 + (106L*x0))];
                    auto tmp5 = in_ptr3[static_cast<long>(x1 + (95L*x0))];
                    auto tmp7 = in_out_ptr0[static_cast<long>(x1 + (84L*x0))];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    in_out_ptr0[static_cast<long>(x1 + (84L*x0))] = tmp8;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(84L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp23 = in_ptr4[static_cast<long>(x0 + (84L*x1))];
                        auto tmp24 = in_ptr5[static_cast<long>(x0)];
                        auto tmp0 = c10::convert<long>(x0);
                        auto tmp1 = static_cast<long>(72);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_out_ptr0[static_cast<long>(x0 + (84L*x1))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = tmp2 ? tmp5 : tmp6;
                        auto tmp8 = tmp0 < tmp1;
                        auto tmp9 = [&]
                        {
                            auto tmp10 = in_out_ptr0[static_cast<long>(x0 + (84L*x1))];
                            return tmp10;
                        }
                        ;
                        auto tmp11 = tmp8 ? tmp9() : static_cast<decltype(tmp9())>(0.0);
                        auto tmp12 = tmp8 ? tmp11 : tmp6;
                        auto tmp13 = decltype(tmp7)(tmp7 + tmp12);
                        auto tmp14 = [&]
                        {
                            auto tmp15 = in_out_ptr0[static_cast<long>(x0 + (84L*x1))];
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp2 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                        auto tmp17 = tmp2 ? tmp16 : tmp6;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = in_out_ptr0[static_cast<long>(x0 + (84L*x1))];
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp8 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                        auto tmp21 = tmp8 ? tmp20 : tmp6;
                        auto tmp22 = decltype(tmp17)(tmp17 + tmp21);
                        auto tmp25 = decltype(tmp23)(tmp23 - tmp24);
                        auto tmp26 = decltype(tmp22)(tmp22 * tmp25);
                        tmp_acc0 = tmp_acc0 + tmp13;
                        tmp_acc1 = tmp_acc1 + tmp26;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(80L); x0<static_cast<long>(84L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    out_ptr2[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(84L); x2+=static_cast<long>(1L))
                    {
                        auto tmp14 = in_ptr4[static_cast<long>(x2 + (84L*x1) + (16464L*x0))];
                        auto tmp15 = in_ptr5[static_cast<long>(x2)];
                        auto tmp17 = out_ptr1[static_cast<long>(x2)];
                        auto tmp20 = in_ptr6[static_cast<long>(x2)];
                        auto tmp25 = out_ptr0[static_cast<long>(x2)];
                        auto tmp28 = in_ptr7[static_cast<long>(x2)];
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(72);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_out_ptr0[static_cast<long>(x2 + (84L*x1) + (16464L*x0))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = tmp2 ? tmp5 : tmp6;
                        auto tmp8 = tmp0 < tmp1;
                        auto tmp9 = [&]
                        {
                            auto tmp10 = in_out_ptr0[static_cast<long>(x2 + (84L*x1) + (16464L*x0))];
                            return tmp10;
                        }
                        ;
                        auto tmp11 = tmp8 ? tmp9() : static_cast<decltype(tmp9())>(0.0);
                        auto tmp12 = tmp8 ? tmp11 : tmp6;
                        auto tmp13 = decltype(tmp7)(tmp7 + tmp12);
                        auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                        auto tmp18 = static_cast<float>(0.0006377551020408163);
                        auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                        auto tmp21 = decltype(tmp20)(tmp20 * tmp20);
                        auto tmp22 = decltype(tmp19)(tmp19 * tmp21);
                        auto tmp23 = decltype(tmp16)(tmp16 * tmp22);
                        auto tmp24 = decltype(tmp13)(tmp13 - tmp23);
                        auto tmp26 = decltype(tmp25)(tmp25 * tmp18);
                        auto tmp27 = decltype(tmp24)(tmp24 - tmp26);
                        auto tmp29 = decltype(tmp20)(tmp20 * tmp28);
                        auto tmp30 = decltype(tmp27)(tmp27 * tmp29);
                        out_ptr3[static_cast<long>(x1 + (196L*x2) + (16464L*x0))] = tmp30;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(432L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (432L*x2) + (84672L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (432L*x0))];
                            auto tmp9 = in_ptr2[static_cast<long>(x1 + (432L*x2) + (84672L*x0))];
                            auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = tmp3 <= tmp4;
                            auto tmp6 = static_cast<float>(6.0);
                            auto tmp7 = tmp3 >= tmp6;
                            auto tmp8 = decltype(tmp5)(tmp5 | tmp7);
                            auto tmp10 = tmp8 ? tmp4 : tmp9;
                            auto tmp11 = decltype(tmp10)(tmp10 * tmp0);
                            tmp_acc0 = tmp_acc0 + tmp11;
                        }
                        out_ptr0[static_cast<long>(x1 + (432L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3456L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_threshold_backward_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (36L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (36L*x1)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (36L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp9 = tmp5 * tmp8;
                    tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    tmp_acc1_vec = tmp_acc1_vec + tmp9;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(32L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (36L*x1))];
                    auto tmp3 = in_ptr1[static_cast<long>(x0 + (36L*x1))];
                    auto tmp5 = in_ptr2[static_cast<long>(x0 + (36L*x1))];
                    auto tmp6 = in_ptr3[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    auto tmp4 = tmp2 ? tmp1 : tmp3;
                    auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                    auto tmp8 = decltype(tmp4)(tmp4 * tmp7);
                    tmp_acc0 = tmp_acc0 + tmp4;
                    tmp_acc1 = tmp_acc1 + tmp8;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(8.0);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 / tmp3;
            auto tmp5 = static_cast<float>(1e-05);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 + tmp6;
            auto tmp8 = tmp7.rsqrt();
            auto tmp9 = tmp0 * tmp8;
            tmp9.store(out_ptr2 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(32L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr4[static_cast<long>(x0)];
            auto tmp2 = static_cast<float>(8.0);
            auto tmp3 = tmp1 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
            auto tmp6 = 1 / std::sqrt(tmp5);
            auto tmp7 = decltype(tmp0)(tmp0 * tmp6);
            out_ptr2[static_cast<long>(x0)] = tmp7;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (36L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (36L*x0)));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (36L*x0)));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp25 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                auto tmp8 = tmp6 - tmp7;
                auto tmp10 = static_cast<float>(0.125);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp14 = static_cast<float>(8.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 / tmp15;
                auto tmp17 = static_cast<float>(1e-05);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 + tmp18;
                auto tmp20 = tmp19.rsqrt();
                auto tmp21 = tmp20 * tmp20;
                auto tmp22 = tmp12 * tmp21;
                auto tmp23 = tmp8 * tmp22;
                auto tmp24 = tmp5 - tmp23;
                auto tmp26 = tmp25 * tmp11;
                auto tmp27 = tmp24 - tmp26;
                auto tmp29 = tmp20 * tmp28;
                auto tmp30 = tmp27 * tmp29;
                tmp30.store(in_out_ptr0 + static_cast<long>(x1 + (36L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(32L); x1<static_cast<long>(36L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (36L*x0))];
                auto tmp3 = in_out_ptr0[static_cast<long>(x1 + (36L*x0))];
                auto tmp5 = in_ptr2[static_cast<long>(x1 + (36L*x0))];
                auto tmp6 = in_ptr3[static_cast<long>(x1)];
                auto tmp8 = out_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr4[static_cast<long>(x1)];
                auto tmp21 = out_ptr0[static_cast<long>(x1)];
                auto tmp24 = in_ptr5[static_cast<long>(x1)];
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp4 = tmp2 ? tmp1 : tmp3;
                auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                auto tmp9 = static_cast<float>(0.125);
                auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                auto tmp12 = static_cast<float>(8.0);
                auto tmp13 = tmp11 / tmp12;
                auto tmp14 = static_cast<float>(1e-05);
                auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                auto tmp16 = 1 / std::sqrt(tmp15);
                auto tmp17 = decltype(tmp16)(tmp16 * tmp16);
                auto tmp18 = decltype(tmp10)(tmp10 * tmp17);
                auto tmp19 = decltype(tmp7)(tmp7 * tmp18);
                auto tmp20 = decltype(tmp4)(tmp4 - tmp19);
                auto tmp22 = decltype(tmp21)(tmp21 * tmp9);
                auto tmp23 = decltype(tmp20)(tmp20 - tmp22);
                auto tmp25 = decltype(tmp16)(tmp16 * tmp24);
                auto tmp26 = decltype(tmp23)(tmp23 * tmp25);
                in_out_ptr0[static_cast<long>(x1 + (36L*x0))] = tmp26;
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0)
{
    auto in_ptr2 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(432L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x0 + (432L*x2) + (84672L*x1))];
                            auto tmp1 = in_ptr1[static_cast<long>(x0 + (432L*x1))];
                            auto tmp9 = in_ptr2[static_cast<long>(x0 + (432L*x2) + (84672L*x1))];
                            auto tmp12 = in_ptr3[static_cast<long>(x0 + (432L*x1))];
                            auto tmp16 = in_ptr4[static_cast<long>(x0 + (432L*x2) + (84672L*x1))];
                            auto tmp17 = in_ptr5[static_cast<long>(x0)];
                            auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = tmp3 <= tmp4;
                            auto tmp6 = static_cast<float>(6.0);
                            auto tmp7 = tmp3 >= tmp6;
                            auto tmp8 = decltype(tmp5)(tmp5 | tmp7);
                            auto tmp10 = tmp8 ? tmp4 : tmp9;
                            auto tmp11 = decltype(tmp10)(tmp10 * tmp2);
                            auto tmp13 = static_cast<float>(196.0);
                            auto tmp14 = tmp12 / tmp13;
                            auto tmp15 = decltype(tmp11)(tmp11 + tmp14);
                            auto tmp18 = decltype(tmp16)(tmp16 - tmp17);
                            auto tmp19 = decltype(tmp15)(tmp15 * tmp18);
                            tmp_acc0 = tmp_acc0 + tmp15;
                            tmp_acc1 = tmp_acc1 + tmp19;
                        }
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(432L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (432L*x1) + (84672L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (432L*x0))];
                        auto tmp9 = in_out_ptr0[static_cast<long>(x2 + (432L*x1) + (84672L*x0))];
                        auto tmp12 = in_ptr3[static_cast<long>(x2 + (432L*x0))];
                        auto tmp16 = in_ptr4[static_cast<long>(x2 + (432L*x1) + (84672L*x0))];
                        auto tmp17 = in_ptr5[static_cast<long>(x2)];
                        auto tmp19 = out_ptr1[static_cast<long>(x2)];
                        auto tmp22 = in_ptr6[static_cast<long>(x2)];
                        auto tmp27 = out_ptr0[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                        auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = tmp3 <= tmp4;
                        auto tmp6 = static_cast<float>(6.0);
                        auto tmp7 = tmp3 >= tmp6;
                        auto tmp8 = decltype(tmp5)(tmp5 | tmp7);
                        auto tmp10 = tmp8 ? tmp4 : tmp9;
                        auto tmp11 = decltype(tmp10)(tmp10 * tmp2);
                        auto tmp13 = static_cast<float>(196.0);
                        auto tmp14 = tmp12 / tmp13;
                        auto tmp15 = decltype(tmp11)(tmp11 + tmp14);
                        auto tmp18 = decltype(tmp16)(tmp16 - tmp17);
                        auto tmp20 = static_cast<float>(0.0006377551020408163);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp23 = decltype(tmp22)(tmp22 * tmp22);
                        auto tmp24 = decltype(tmp21)(tmp21 * tmp23);
                        auto tmp25 = decltype(tmp18)(tmp18 * tmp24);
                        auto tmp26 = decltype(tmp15)(tmp15 - tmp25);
                        auto tmp28 = decltype(tmp27)(tmp27 * tmp20);
                        auto tmp29 = decltype(tmp26)(tmp26 - tmp28);
                        in_out_ptr0[static_cast<long>(x2 + (432L*x1) + (84672L*x0))] = tmp29;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(432L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(432L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (432L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (432L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(432L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (432L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (432L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (432L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(432L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(432L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (432L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (432L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (432L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.0006377551020408163);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (432L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (84L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (72L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (72L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp6 = tmp2 * tmp5;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    tmp_acc1_vec = tmp_acc1_vec + tmp6;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (84L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (72L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp5 = tmp3 - tmp4;
                auto tmp7 = static_cast<float>(0.0006377551020408163);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp11 = tmp10 * tmp10;
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = tmp5 * tmp12;
                auto tmp14 = tmp2 - tmp13;
                auto tmp16 = tmp15 * tmp8;
                auto tmp17 = tmp14 - tmp16;
                auto tmp19 = tmp10 * tmp18;
                auto tmp20 = tmp17 * tmp19;
                tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (72L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(366L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (366L*x2) + (71736L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (366L*x0))];
                            auto tmp9 = in_ptr2[static_cast<long>(x1 + (366L*x2) + (71736L*x0))];
                            auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = tmp3 <= tmp4;
                            auto tmp6 = static_cast<float>(6.0);
                            auto tmp7 = tmp3 >= tmp6;
                            auto tmp8 = decltype(tmp5)(tmp5 | tmp7);
                            auto tmp10 = tmp8 ? tmp4 : tmp9;
                            auto tmp11 = decltype(tmp10)(tmp10 * tmp0);
                            tmp_acc0 = tmp_acc0 + tmp11;
                        }
                        out_ptr0[static_cast<long>(x1 + (366L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2928L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_threshold_backward_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (30L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (30L*x1)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (30L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp9 = tmp5 * tmp8;
                    tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    tmp_acc1_vec = tmp_acc1_vec + tmp9;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(24L); x0<static_cast<long>(30L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (30L*x1))];
                    auto tmp3 = in_ptr1[static_cast<long>(x0 + (30L*x1))];
                    auto tmp5 = in_ptr2[static_cast<long>(x0 + (30L*x1))];
                    auto tmp6 = in_ptr3[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    auto tmp4 = tmp2 ? tmp1 : tmp3;
                    auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                    auto tmp8 = decltype(tmp4)(tmp4 * tmp7);
                    tmp_acc0 = tmp_acc0 + tmp4;
                    tmp_acc1 = tmp_acc1 + tmp8;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(8.0);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 / tmp3;
            auto tmp5 = static_cast<float>(1e-05);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 + tmp6;
            auto tmp8 = tmp7.rsqrt();
            auto tmp9 = tmp0 * tmp8;
            tmp9.store(out_ptr2 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(24L); x0<static_cast<long>(30L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr4[static_cast<long>(x0)];
            auto tmp2 = static_cast<float>(8.0);
            auto tmp3 = tmp1 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
            auto tmp6 = 1 / std::sqrt(tmp5);
            auto tmp7 = decltype(tmp0)(tmp0 * tmp6);
            out_ptr2[static_cast<long>(x0)] = tmp7;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (30L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (30L*x0)));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (30L*x0)));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp25 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                auto tmp8 = tmp6 - tmp7;
                auto tmp10 = static_cast<float>(0.125);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp14 = static_cast<float>(8.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 / tmp15;
                auto tmp17 = static_cast<float>(1e-05);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 + tmp18;
                auto tmp20 = tmp19.rsqrt();
                auto tmp21 = tmp20 * tmp20;
                auto tmp22 = tmp12 * tmp21;
                auto tmp23 = tmp8 * tmp22;
                auto tmp24 = tmp5 - tmp23;
                auto tmp26 = tmp25 * tmp11;
                auto tmp27 = tmp24 - tmp26;
                auto tmp29 = tmp20 * tmp28;
                auto tmp30 = tmp27 * tmp29;
                tmp30.store(in_out_ptr0 + static_cast<long>(x1 + (30L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(24L); x1<static_cast<long>(30L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (30L*x0))];
                auto tmp3 = in_out_ptr0[static_cast<long>(x1 + (30L*x0))];
                auto tmp5 = in_ptr2[static_cast<long>(x1 + (30L*x0))];
                auto tmp6 = in_ptr3[static_cast<long>(x1)];
                auto tmp8 = out_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr4[static_cast<long>(x1)];
                auto tmp21 = out_ptr0[static_cast<long>(x1)];
                auto tmp24 = in_ptr5[static_cast<long>(x1)];
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp4 = tmp2 ? tmp1 : tmp3;
                auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                auto tmp9 = static_cast<float>(0.125);
                auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                auto tmp12 = static_cast<float>(8.0);
                auto tmp13 = tmp11 / tmp12;
                auto tmp14 = static_cast<float>(1e-05);
                auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                auto tmp16 = 1 / std::sqrt(tmp15);
                auto tmp17 = decltype(tmp16)(tmp16 * tmp16);
                auto tmp18 = decltype(tmp10)(tmp10 * tmp17);
                auto tmp19 = decltype(tmp7)(tmp7 * tmp18);
                auto tmp20 = decltype(tmp4)(tmp4 - tmp19);
                auto tmp22 = decltype(tmp21)(tmp21 * tmp9);
                auto tmp23 = decltype(tmp20)(tmp20 - tmp22);
                auto tmp25 = decltype(tmp16)(tmp16 * tmp24);
                auto tmp26 = decltype(tmp23)(tmp23 * tmp25);
                in_out_ptr0[static_cast<long>(x1 + (30L*x0))] = tmp26;
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0)
{
    auto in_ptr2 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(366L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x0 + (366L*x2) + (71736L*x1))];
                            auto tmp1 = in_ptr1[static_cast<long>(x0 + (366L*x1))];
                            auto tmp9 = in_ptr2[static_cast<long>(x0 + (366L*x2) + (71736L*x1))];
                            auto tmp12 = in_ptr3[static_cast<long>(x0 + (366L*x1))];
                            auto tmp16 = in_ptr4[static_cast<long>(x0 + (366L*x2) + (71736L*x1))];
                            auto tmp17 = in_ptr5[static_cast<long>(x0)];
                            auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = tmp3 <= tmp4;
                            auto tmp6 = static_cast<float>(6.0);
                            auto tmp7 = tmp3 >= tmp6;
                            auto tmp8 = decltype(tmp5)(tmp5 | tmp7);
                            auto tmp10 = tmp8 ? tmp4 : tmp9;
                            auto tmp11 = decltype(tmp10)(tmp10 * tmp2);
                            auto tmp13 = static_cast<float>(196.0);
                            auto tmp14 = tmp12 / tmp13;
                            auto tmp15 = decltype(tmp11)(tmp11 + tmp14);
                            auto tmp18 = decltype(tmp16)(tmp16 - tmp17);
                            auto tmp19 = decltype(tmp15)(tmp15 * tmp18);
                            tmp_acc0 = tmp_acc0 + tmp15;
                            tmp_acc1 = tmp_acc1 + tmp19;
                        }
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(366L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (366L*x1) + (71736L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (366L*x0))];
                        auto tmp9 = in_out_ptr0[static_cast<long>(x2 + (366L*x1) + (71736L*x0))];
                        auto tmp12 = in_ptr3[static_cast<long>(x2 + (366L*x0))];
                        auto tmp16 = in_ptr4[static_cast<long>(x2 + (366L*x1) + (71736L*x0))];
                        auto tmp17 = in_ptr5[static_cast<long>(x2)];
                        auto tmp19 = out_ptr1[static_cast<long>(x2)];
                        auto tmp22 = in_ptr6[static_cast<long>(x2)];
                        auto tmp27 = out_ptr0[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                        auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = tmp3 <= tmp4;
                        auto tmp6 = static_cast<float>(6.0);
                        auto tmp7 = tmp3 >= tmp6;
                        auto tmp8 = decltype(tmp5)(tmp5 | tmp7);
                        auto tmp10 = tmp8 ? tmp4 : tmp9;
                        auto tmp11 = decltype(tmp10)(tmp10 * tmp2);
                        auto tmp13 = static_cast<float>(196.0);
                        auto tmp14 = tmp12 / tmp13;
                        auto tmp15 = decltype(tmp11)(tmp11 + tmp14);
                        auto tmp18 = decltype(tmp16)(tmp16 - tmp17);
                        auto tmp20 = static_cast<float>(0.0006377551020408163);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp23 = decltype(tmp22)(tmp22 * tmp22);
                        auto tmp24 = decltype(tmp21)(tmp21 * tmp23);
                        auto tmp25 = decltype(tmp18)(tmp18 * tmp24);
                        auto tmp26 = decltype(tmp15)(tmp15 - tmp25);
                        auto tmp28 = decltype(tmp27)(tmp27 * tmp20);
                        auto tmp29 = decltype(tmp26)(tmp26 - tmp28);
                        in_out_ptr0[static_cast<long>(x2 + (366L*x1) + (71736L*x0))] = tmp29;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(360L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(360L); x0<static_cast<long>(366L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    in_out_ptr1[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (366L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (366L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(360L); x1<static_cast<long>(366L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (366L*x0))];
                    auto tmp1 = in_ptr6[static_cast<long>(x1)];
                    auto tmp2 = in_ptr7[static_cast<long>(x1)];
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                    in_out_ptr0[static_cast<long>(x1 + (366L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(360L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (366L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (366L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (366L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(360L); x0<static_cast<long>(366L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (366L*x1))];
                        auto tmp1 = in_ptr1[static_cast<long>(x0 + (366L*x1))];
                        auto tmp3 = in_ptr2[static_cast<long>(x0 + (366L*x1))];
                        auto tmp4 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = decltype(tmp2)(tmp2 * tmp5);
                        tmp_acc0 = tmp_acc0 + tmp2;
                        tmp_acc1 = tmp_acc1 + tmp6;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(360L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(360L); x0<static_cast<long>(366L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    out_ptr2[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(360L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (366L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (366L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (366L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.00015943877551020407);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (366L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(360L); x1<static_cast<long>(366L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (366L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (366L*x0))];
                    auto tmp3 = in_ptr2[static_cast<long>(x1 + (366L*x0))];
                    auto tmp4 = in_ptr3[static_cast<long>(x1)];
                    auto tmp6 = out_ptr1[static_cast<long>(x1)];
                    auto tmp9 = in_ptr4[static_cast<long>(x1)];
                    auto tmp14 = out_ptr0[static_cast<long>(x1)];
                    auto tmp17 = in_ptr5[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                    auto tmp7 = static_cast<float>(0.00015943877551020407);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp10 = decltype(tmp9)(tmp9 * tmp9);
                    auto tmp11 = decltype(tmp8)(tmp8 * tmp10);
                    auto tmp12 = decltype(tmp5)(tmp5 * tmp11);
                    auto tmp13 = decltype(tmp2)(tmp2 - tmp12);
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp7);
                    auto tmp16 = decltype(tmp13)(tmp13 - tmp15);
                    auto tmp18 = decltype(tmp9)(tmp9 * tmp17);
                    auto tmp19 = decltype(tmp16)(tmp16 * tmp18);
                    in_out_ptr0[static_cast<long>(x1 + (366L*x0))] = tmp19;
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_57 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(61L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp23 = in_ptr1[static_cast<long>(x0 + (61L*x1))];
                        auto tmp24 = in_ptr2[static_cast<long>(x0)];
                        auto tmp0 = c10::convert<long>(x0);
                        auto tmp1 = static_cast<long>(50);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr0[static_cast<long>(x0 + (61L*x1))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = tmp2 ? tmp5 : tmp6;
                        auto tmp8 = tmp0 < tmp1;
                        auto tmp9 = [&]
                        {
                            auto tmp10 = in_ptr0[static_cast<long>(x0 + (61L*x1))];
                            return tmp10;
                        }
                        ;
                        auto tmp11 = tmp8 ? tmp9() : static_cast<decltype(tmp9())>(0.0);
                        auto tmp12 = tmp8 ? tmp11 : tmp6;
                        auto tmp13 = decltype(tmp7)(tmp7 + tmp12);
                        auto tmp14 = [&]
                        {
                            auto tmp15 = in_ptr0[static_cast<long>(x0 + (61L*x1))];
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp2 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                        auto tmp17 = tmp2 ? tmp16 : tmp6;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = in_ptr0[static_cast<long>(x0 + (61L*x1))];
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp8 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                        auto tmp21 = tmp8 ? tmp20 : tmp6;
                        auto tmp22 = decltype(tmp17)(tmp17 + tmp21);
                        auto tmp25 = decltype(tmp23)(tmp23 - tmp24);
                        auto tmp26 = decltype(tmp22)(tmp22 * tmp25);
                        tmp_acc0 = tmp_acc0 + tmp13;
                        tmp_acc1 = tmp_acc1 + tmp26;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(56L); x0<static_cast<long>(61L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    out_ptr2[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(61L); x2+=static_cast<long>(1L))
                    {
                        auto tmp14 = in_ptr1[static_cast<long>(x2 + (61L*x1) + (47824L*x0))];
                        auto tmp15 = in_ptr2[static_cast<long>(x2)];
                        auto tmp17 = out_ptr1[static_cast<long>(x2)];
                        auto tmp20 = in_ptr3[static_cast<long>(x2)];
                        auto tmp25 = out_ptr0[static_cast<long>(x2)];
                        auto tmp28 = in_ptr4[static_cast<long>(x2)];
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(50);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr0[static_cast<long>(x2 + (61L*x1) + (47824L*x0))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = tmp2 ? tmp5 : tmp6;
                        auto tmp8 = tmp0 < tmp1;
                        auto tmp9 = [&]
                        {
                            auto tmp10 = in_ptr0[static_cast<long>(x2 + (61L*x1) + (47824L*x0))];
                            return tmp10;
                        }
                        ;
                        auto tmp11 = tmp8 ? tmp9() : static_cast<decltype(tmp9())>(0.0);
                        auto tmp12 = tmp8 ? tmp11 : tmp6;
                        auto tmp13 = decltype(tmp7)(tmp7 + tmp12);
                        auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                        auto tmp18 = static_cast<float>(0.00015943877551020407);
                        auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                        auto tmp21 = decltype(tmp20)(tmp20 * tmp20);
                        auto tmp22 = decltype(tmp19)(tmp19 * tmp21);
                        auto tmp23 = decltype(tmp16)(tmp16 * tmp22);
                        auto tmp24 = decltype(tmp13)(tmp13 - tmp23);
                        auto tmp26 = decltype(tmp25)(tmp25 * tmp18);
                        auto tmp27 = decltype(tmp24)(tmp24 - tmp26);
                        auto tmp29 = decltype(tmp20)(tmp20 * tmp28);
                        auto tmp30 = decltype(tmp27)(tmp27 * tmp29);
                        out_ptr3[static_cast<long>(x1 + (784L*x2) + (47824L*x0))] = tmp30;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(300L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (300L*x2) + (235200L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (300L*x0))];
                            auto tmp9 = in_ptr2[static_cast<long>(x1 + (300L*x2) + (235200L*x0))];
                            auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = tmp3 <= tmp4;
                            auto tmp6 = static_cast<float>(6.0);
                            auto tmp7 = tmp3 >= tmp6;
                            auto tmp8 = decltype(tmp5)(tmp5 | tmp7);
                            auto tmp10 = tmp8 ? tmp4 : tmp9;
                            auto tmp11 = decltype(tmp10)(tmp10 * tmp0);
                            tmp_acc0 = tmp_acc0 + tmp11;
                        }
                        out_ptr0[static_cast<long>(x1 + (300L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2400L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_threshold_backward_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (25L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (25L*x1)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (25L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp9 = tmp5 * tmp8;
                    tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    tmp_acc1_vec = tmp_acc1_vec + tmp9;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(24L); x0<static_cast<long>(25L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (25L*x1))];
                    auto tmp3 = in_ptr1[static_cast<long>(x0 + (25L*x1))];
                    auto tmp5 = in_ptr2[static_cast<long>(x0 + (25L*x1))];
                    auto tmp6 = in_ptr3[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    auto tmp4 = tmp2 ? tmp1 : tmp3;
                    auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                    auto tmp8 = decltype(tmp4)(tmp4 * tmp7);
                    tmp_acc0 = tmp_acc0 + tmp4;
                    tmp_acc1 = tmp_acc1 + tmp8;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(8.0);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 / tmp3;
            auto tmp5 = static_cast<float>(1e-05);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 + tmp6;
            auto tmp8 = tmp7.rsqrt();
            auto tmp9 = tmp0 * tmp8;
            tmp9.store(out_ptr2 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(24L); x0<static_cast<long>(25L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr4[static_cast<long>(x0)];
            auto tmp2 = static_cast<float>(8.0);
            auto tmp3 = tmp1 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
            auto tmp6 = 1 / std::sqrt(tmp5);
            auto tmp7 = decltype(tmp0)(tmp0 * tmp6);
            out_ptr2[static_cast<long>(x0)] = tmp7;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (25L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (25L*x0)));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (25L*x0)));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp25 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                auto tmp8 = tmp6 - tmp7;
                auto tmp10 = static_cast<float>(0.125);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp14 = static_cast<float>(8.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 / tmp15;
                auto tmp17 = static_cast<float>(1e-05);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 + tmp18;
                auto tmp20 = tmp19.rsqrt();
                auto tmp21 = tmp20 * tmp20;
                auto tmp22 = tmp12 * tmp21;
                auto tmp23 = tmp8 * tmp22;
                auto tmp24 = tmp5 - tmp23;
                auto tmp26 = tmp25 * tmp11;
                auto tmp27 = tmp24 - tmp26;
                auto tmp29 = tmp20 * tmp28;
                auto tmp30 = tmp27 * tmp29;
                tmp30.store(in_out_ptr0 + static_cast<long>(x1 + (25L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(24L); x1<static_cast<long>(25L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (25L*x0))];
                auto tmp3 = in_out_ptr0[static_cast<long>(x1 + (25L*x0))];
                auto tmp5 = in_ptr2[static_cast<long>(x1 + (25L*x0))];
                auto tmp6 = in_ptr3[static_cast<long>(x1)];
                auto tmp8 = out_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr4[static_cast<long>(x1)];
                auto tmp21 = out_ptr0[static_cast<long>(x1)];
                auto tmp24 = in_ptr5[static_cast<long>(x1)];
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp4 = tmp2 ? tmp1 : tmp3;
                auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                auto tmp9 = static_cast<float>(0.125);
                auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                auto tmp12 = static_cast<float>(8.0);
                auto tmp13 = tmp11 / tmp12;
                auto tmp14 = static_cast<float>(1e-05);
                auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                auto tmp16 = 1 / std::sqrt(tmp15);
                auto tmp17 = decltype(tmp16)(tmp16 * tmp16);
                auto tmp18 = decltype(tmp10)(tmp10 * tmp17);
                auto tmp19 = decltype(tmp7)(tmp7 * tmp18);
                auto tmp20 = decltype(tmp4)(tmp4 - tmp19);
                auto tmp22 = decltype(tmp21)(tmp21 * tmp9);
                auto tmp23 = decltype(tmp20)(tmp20 - tmp22);
                auto tmp25 = decltype(tmp16)(tmp16 * tmp24);
                auto tmp26 = decltype(tmp23)(tmp23 * tmp25);
                in_out_ptr0[static_cast<long>(x1 + (25L*x0))] = tmp26;
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0)
{
    auto in_ptr2 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(300L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x0 + (300L*x2) + (235200L*x1))];
                            auto tmp1 = in_ptr1[static_cast<long>(x0 + (300L*x1))];
                            auto tmp9 = in_ptr2[static_cast<long>(x0 + (300L*x2) + (235200L*x1))];
                            auto tmp12 = in_ptr3[static_cast<long>(x0 + (300L*x1))];
                            auto tmp16 = in_ptr4[static_cast<long>(x0 + (300L*x2) + (235200L*x1))];
                            auto tmp17 = in_ptr5[static_cast<long>(x0)];
                            auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = tmp3 <= tmp4;
                            auto tmp6 = static_cast<float>(6.0);
                            auto tmp7 = tmp3 >= tmp6;
                            auto tmp8 = decltype(tmp5)(tmp5 | tmp7);
                            auto tmp10 = tmp8 ? tmp4 : tmp9;
                            auto tmp11 = decltype(tmp10)(tmp10 * tmp2);
                            auto tmp13 = static_cast<float>(784.0);
                            auto tmp14 = tmp12 / tmp13;
                            auto tmp15 = decltype(tmp11)(tmp11 + tmp14);
                            auto tmp18 = decltype(tmp16)(tmp16 - tmp17);
                            auto tmp19 = decltype(tmp15)(tmp15 * tmp18);
                            tmp_acc0 = tmp_acc0 + tmp15;
                            tmp_acc1 = tmp_acc1 + tmp19;
                        }
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(300L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (300L*x1) + (235200L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (300L*x0))];
                        auto tmp9 = in_out_ptr0[static_cast<long>(x2 + (300L*x1) + (235200L*x0))];
                        auto tmp12 = in_ptr3[static_cast<long>(x2 + (300L*x0))];
                        auto tmp16 = in_ptr4[static_cast<long>(x2 + (300L*x1) + (235200L*x0))];
                        auto tmp17 = in_ptr5[static_cast<long>(x2)];
                        auto tmp19 = out_ptr1[static_cast<long>(x2)];
                        auto tmp22 = in_ptr6[static_cast<long>(x2)];
                        auto tmp27 = out_ptr0[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                        auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = tmp3 <= tmp4;
                        auto tmp6 = static_cast<float>(6.0);
                        auto tmp7 = tmp3 >= tmp6;
                        auto tmp8 = decltype(tmp5)(tmp5 | tmp7);
                        auto tmp10 = tmp8 ? tmp4 : tmp9;
                        auto tmp11 = decltype(tmp10)(tmp10 * tmp2);
                        auto tmp13 = static_cast<float>(784.0);
                        auto tmp14 = tmp12 / tmp13;
                        auto tmp15 = decltype(tmp11)(tmp11 + tmp14);
                        auto tmp18 = decltype(tmp16)(tmp16 - tmp17);
                        auto tmp20 = static_cast<float>(0.00015943877551020407);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp23 = decltype(tmp22)(tmp22 * tmp22);
                        auto tmp24 = decltype(tmp21)(tmp21 * tmp23);
                        auto tmp25 = decltype(tmp18)(tmp18 * tmp24);
                        auto tmp26 = decltype(tmp15)(tmp15 - tmp25);
                        auto tmp28 = decltype(tmp27)(tmp27 * tmp20);
                        auto tmp29 = decltype(tmp26)(tmp26 - tmp28);
                        in_out_ptr0[static_cast<long>(x2 + (300L*x1) + (235200L*x0))] = tmp29;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(296L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(296L); x0<static_cast<long>(300L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    in_out_ptr1[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(296L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (300L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (300L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(296L); x1<static_cast<long>(300L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (300L*x0))];
                    auto tmp1 = in_ptr6[static_cast<long>(x1)];
                    auto tmp2 = in_ptr7[static_cast<long>(x1)];
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                    in_out_ptr0[static_cast<long>(x1 + (300L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(296L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (300L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (300L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (300L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(296L); x0<static_cast<long>(300L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (300L*x1))];
                        auto tmp1 = in_ptr1[static_cast<long>(x0 + (300L*x1))];
                        auto tmp3 = in_ptr2[static_cast<long>(x0 + (300L*x1))];
                        auto tmp4 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = decltype(tmp2)(tmp2 * tmp5);
                        tmp_acc0 = tmp_acc0 + tmp2;
                        tmp_acc1 = tmp_acc1 + tmp6;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(296L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(296L); x0<static_cast<long>(300L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    out_ptr2[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(296L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (300L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (300L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (300L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.00015943877551020407);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (300L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(296L); x1<static_cast<long>(300L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (300L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (300L*x0))];
                    auto tmp3 = in_ptr2[static_cast<long>(x1 + (300L*x0))];
                    auto tmp4 = in_ptr3[static_cast<long>(x1)];
                    auto tmp6 = out_ptr1[static_cast<long>(x1)];
                    auto tmp9 = in_ptr4[static_cast<long>(x1)];
                    auto tmp14 = out_ptr0[static_cast<long>(x1)];
                    auto tmp17 = in_ptr5[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                    auto tmp7 = static_cast<float>(0.00015943877551020407);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp10 = decltype(tmp9)(tmp9 * tmp9);
                    auto tmp11 = decltype(tmp8)(tmp8 * tmp10);
                    auto tmp12 = decltype(tmp5)(tmp5 * tmp11);
                    auto tmp13 = decltype(tmp2)(tmp2 - tmp12);
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp7);
                    auto tmp16 = decltype(tmp13)(tmp13 - tmp15);
                    auto tmp18 = decltype(tmp9)(tmp9 * tmp17);
                    auto tmp19 = decltype(tmp16)(tmp16 * tmp18);
                    in_out_ptr0[static_cast<long>(x1 + (300L*x0))] = tmp19;
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (61L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (50L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (50L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(48L); x0<static_cast<long>(50L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (61L*x1))];
                        auto tmp1 = in_ptr1[static_cast<long>(x0 + (50L*x1))];
                        auto tmp3 = in_ptr2[static_cast<long>(x0 + (50L*x1))];
                        auto tmp4 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = decltype(tmp2)(tmp2 * tmp5);
                        tmp_acc0 = tmp_acc0 + tmp2;
                        tmp_acc1 = tmp_acc1 + tmp6;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(48L); x0<static_cast<long>(50L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    out_ptr2[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (61L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (50L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (50L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(0.00015943877551020407);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (50L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(50L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (61L*x0))];
                    auto tmp1 = in_out_ptr0[static_cast<long>(x1 + (50L*x0))];
                    auto tmp3 = in_ptr2[static_cast<long>(x1 + (50L*x0))];
                    auto tmp4 = in_ptr3[static_cast<long>(x1)];
                    auto tmp6 = out_ptr1[static_cast<long>(x1)];
                    auto tmp9 = in_ptr4[static_cast<long>(x1)];
                    auto tmp14 = out_ptr0[static_cast<long>(x1)];
                    auto tmp17 = in_ptr5[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                    auto tmp7 = static_cast<float>(0.00015943877551020407);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp10 = decltype(tmp9)(tmp9 * tmp9);
                    auto tmp11 = decltype(tmp8)(tmp8 * tmp10);
                    auto tmp12 = decltype(tmp5)(tmp5 * tmp11);
                    auto tmp13 = decltype(tmp2)(tmp2 - tmp12);
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp7);
                    auto tmp16 = decltype(tmp13)(tmp13 - tmp15);
                    auto tmp18 = decltype(tmp9)(tmp9 * tmp17);
                    auto tmp19 = decltype(tmp16)(tmp16 * tmp18);
                    in_out_ptr0[static_cast<long>(x1 + (50L*x0))] = tmp19;
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(228L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = 0;
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (228L*x2) + (178752L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (228L*x0))];
                            auto tmp9 = in_ptr2[static_cast<long>(x1 + (228L*x2) + (178752L*x0))];
                            auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = tmp3 <= tmp4;
                            auto tmp6 = static_cast<float>(6.0);
                            auto tmp7 = tmp3 >= tmp6;
                            auto tmp8 = decltype(tmp5)(tmp5 | tmp7);
                            auto tmp10 = tmp8 ? tmp4 : tmp9;
                            auto tmp11 = decltype(tmp10)(tmp10 * tmp0);
                            tmp_acc0 = tmp_acc0 + tmp11;
                        }
                        out_ptr0[static_cast<long>(x1 + (228L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1824L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_threshold_backward_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (19L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (19L*x1)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (19L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp9 = tmp5 * tmp8;
                    tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    tmp_acc1_vec = tmp_acc1_vec + tmp9;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
            }
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(16L); x0<static_cast<long>(19L); x0+=static_cast<long>(1L))
        {
            {
                float tmp_acc0 = 0;
                float tmp_acc1 = 0;
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0 + (19L*x1))];
                    auto tmp3 = in_ptr1[static_cast<long>(x0 + (19L*x1))];
                    auto tmp5 = in_ptr2[static_cast<long>(x0 + (19L*x1))];
                    auto tmp6 = in_ptr3[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = tmp0 <= tmp1;
                    auto tmp4 = tmp2 ? tmp1 : tmp3;
                    auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                    auto tmp8 = decltype(tmp4)(tmp4 * tmp7);
                    tmp_acc0 = tmp_acc0 + tmp4;
                    tmp_acc1 = tmp_acc1 + tmp8;
                }
                out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                out_ptr1[static_cast<long>(x0)] = tmp_acc1;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(8.0);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 / tmp3;
            auto tmp5 = static_cast<float>(1e-05);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 + tmp6;
            auto tmp8 = tmp7.rsqrt();
            auto tmp9 = tmp0 * tmp8;
            tmp9.store(out_ptr2 + static_cast<long>(x0));
        }
        #pragma omp simd simdlen(4) 
        for(long x0=static_cast<long>(16L); x0<static_cast<long>(19L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr1[static_cast<long>(x0)];
            auto tmp1 = in_ptr4[static_cast<long>(x0)];
            auto tmp2 = static_cast<float>(8.0);
            auto tmp3 = tmp1 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
            auto tmp6 = 1 / std::sqrt(tmp5);
            auto tmp7 = decltype(tmp0)(tmp0 * tmp6);
            out_ptr2[static_cast<long>(x0)] = tmp7;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (19L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (19L*x0)));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (19L*x0)));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp25 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                auto tmp8 = tmp6 - tmp7;
                auto tmp10 = static_cast<float>(0.125);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp14 = static_cast<float>(8.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 / tmp15;
                auto tmp17 = static_cast<float>(1e-05);
                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                auto tmp19 = tmp16 + tmp18;
                auto tmp20 = tmp19.rsqrt();
                auto tmp21 = tmp20 * tmp20;
                auto tmp22 = tmp12 * tmp21;
                auto tmp23 = tmp8 * tmp22;
                auto tmp24 = tmp5 - tmp23;
                auto tmp26 = tmp25 * tmp11;
                auto tmp27 = tmp24 - tmp26;
                auto tmp29 = tmp20 * tmp28;
                auto tmp30 = tmp27 * tmp29;
                tmp30.store(in_out_ptr0 + static_cast<long>(x1 + (19L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(16L); x1<static_cast<long>(19L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1 + (19L*x0))];
                auto tmp3 = in_out_ptr0[static_cast<long>(x1 + (19L*x0))];
                auto tmp5 = in_ptr2[static_cast<long>(x1 + (19L*x0))];
                auto tmp6 = in_ptr3[static_cast<long>(x1)];
                auto tmp8 = out_ptr1[static_cast<long>(x1)];
                auto tmp11 = in_ptr4[static_cast<long>(x1)];
                auto tmp21 = out_ptr0[static_cast<long>(x1)];
                auto tmp24 = in_ptr5[static_cast<long>(x1)];
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = tmp0 <= tmp1;
                auto tmp4 = tmp2 ? tmp1 : tmp3;
                auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                auto tmp9 = static_cast<float>(0.125);
                auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                auto tmp12 = static_cast<float>(8.0);
                auto tmp13 = tmp11 / tmp12;
                auto tmp14 = static_cast<float>(1e-05);
                auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                auto tmp16 = 1 / std::sqrt(tmp15);
                auto tmp17 = decltype(tmp16)(tmp16 * tmp16);
                auto tmp18 = decltype(tmp10)(tmp10 * tmp17);
                auto tmp19 = decltype(tmp7)(tmp7 * tmp18);
                auto tmp20 = decltype(tmp4)(tmp4 - tmp19);
                auto tmp22 = decltype(tmp21)(tmp21 * tmp9);
                auto tmp23 = decltype(tmp20)(tmp20 - tmp22);
                auto tmp25 = decltype(tmp16)(tmp16 * tmp24);
                auto tmp26 = decltype(tmp23)(tmp23 * tmp25);
                in_out_ptr0[static_cast<long>(x1 + (19L*x0))] = tmp26;
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0)
{
    auto in_ptr2 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(228L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x0 + (228L*x2) + (178752L*x1))];
                            auto tmp1 = in_ptr1[static_cast<long>(x0 + (228L*x1))];
                            auto tmp9 = in_ptr2[static_cast<long>(x0 + (228L*x2) + (178752L*x1))];
                            auto tmp12 = in_ptr3[static_cast<long>(x0 + (228L*x1))];
                            auto tmp16 = in_ptr4[static_cast<long>(x0 + (228L*x2) + (178752L*x1))];
                            auto tmp17 = in_ptr5[static_cast<long>(x0)];
                            auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                            auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = tmp3 <= tmp4;
                            auto tmp6 = static_cast<float>(6.0);
                            auto tmp7 = tmp3 >= tmp6;
                            auto tmp8 = decltype(tmp5)(tmp5 | tmp7);
                            auto tmp10 = tmp8 ? tmp4 : tmp9;
                            auto tmp11 = decltype(tmp10)(tmp10 * tmp2);
                            auto tmp13 = static_cast<float>(784.0);
                            auto tmp14 = tmp12 / tmp13;
                            auto tmp15 = decltype(tmp11)(tmp11 + tmp14);
                            auto tmp18 = decltype(tmp16)(tmp16 - tmp17);
                            auto tmp19 = decltype(tmp15)(tmp15 * tmp18);
                            tmp_acc0 = tmp_acc0 + tmp15;
                            tmp_acc1 = tmp_acc1 + tmp19;
                        }
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(228L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (228L*x1) + (178752L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (228L*x0))];
                        auto tmp9 = in_out_ptr0[static_cast<long>(x2 + (228L*x1) + (178752L*x0))];
                        auto tmp12 = in_ptr3[static_cast<long>(x2 + (228L*x0))];
                        auto tmp16 = in_ptr4[static_cast<long>(x2 + (228L*x1) + (178752L*x0))];
                        auto tmp17 = in_ptr5[static_cast<long>(x2)];
                        auto tmp19 = out_ptr1[static_cast<long>(x2)];
                        auto tmp22 = in_ptr6[static_cast<long>(x2)];
                        auto tmp27 = out_ptr0[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp1)(1) / (decltype(tmp1)(1) + std::exp(-tmp1));
                        auto tmp3 = decltype(tmp0)(tmp0 * tmp2);
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = tmp3 <= tmp4;
                        auto tmp6 = static_cast<float>(6.0);
                        auto tmp7 = tmp3 >= tmp6;
                        auto tmp8 = decltype(tmp5)(tmp5 | tmp7);
                        auto tmp10 = tmp8 ? tmp4 : tmp9;
                        auto tmp11 = decltype(tmp10)(tmp10 * tmp2);
                        auto tmp13 = static_cast<float>(784.0);
                        auto tmp14 = tmp12 / tmp13;
                        auto tmp15 = decltype(tmp11)(tmp11 + tmp14);
                        auto tmp18 = decltype(tmp16)(tmp16 - tmp17);
                        auto tmp20 = static_cast<float>(0.00015943877551020407);
                        auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                        auto tmp23 = decltype(tmp22)(tmp22 * tmp22);
                        auto tmp24 = decltype(tmp21)(tmp21 * tmp23);
                        auto tmp25 = decltype(tmp18)(tmp18 * tmp24);
                        auto tmp26 = decltype(tmp15)(tmp15 - tmp25);
                        auto tmp28 = decltype(tmp27)(tmp27 * tmp20);
                        auto tmp29 = decltype(tmp26)(tmp26 - tmp28);
                        in_out_ptr0[static_cast<long>(x2 + (228L*x1) + (178752L*x0))] = tmp29;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(224L); x0<static_cast<long>(228L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    in_out_ptr1[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (228L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 * tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (228L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(224L); x1<static_cast<long>(228L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (228L*x0))];
                    auto tmp1 = in_ptr6[static_cast<long>(x1)];
                    auto tmp2 = in_ptr7[static_cast<long>(x1)];
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = decltype(tmp0)(tmp0 * tmp3);
                    in_out_ptr0[static_cast<long>(x1 + (228L*x0))] = tmp4;
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (228L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (228L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (228L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(224L); x0<static_cast<long>(228L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (228L*x1))];
                        auto tmp1 = in_ptr1[static_cast<long>(x0 + (228L*x1))];
                        auto tmp3 = in_ptr2[static_cast<long>(x0 + (228L*x1))];
                        auto tmp4 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = decltype(tmp2)(tmp2 * tmp5);
                        tmp_acc0 = tmp_acc0 + tmp2;
                        tmp_acc1 = tmp_acc1 + tmp6;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(224L); x0<static_cast<long>(228L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    out_ptr2[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (228L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (228L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (228L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(3.985969387755102e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (228L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(224L); x1<static_cast<long>(228L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (228L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (228L*x0))];
                    auto tmp3 = in_ptr2[static_cast<long>(x1 + (228L*x0))];
                    auto tmp4 = in_ptr3[static_cast<long>(x1)];
                    auto tmp6 = out_ptr1[static_cast<long>(x1)];
                    auto tmp9 = in_ptr4[static_cast<long>(x1)];
                    auto tmp14 = out_ptr0[static_cast<long>(x1)];
                    auto tmp17 = in_ptr5[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                    auto tmp7 = static_cast<float>(3.985969387755102e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp10 = decltype(tmp9)(tmp9 * tmp9);
                    auto tmp11 = decltype(tmp8)(tmp8 * tmp10);
                    auto tmp12 = decltype(tmp5)(tmp5 * tmp11);
                    auto tmp13 = decltype(tmp2)(tmp2 - tmp12);
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp7);
                    auto tmp16 = decltype(tmp13)(tmp13 - tmp15);
                    auto tmp18 = decltype(tmp9)(tmp9 * tmp17);
                    auto tmp19 = decltype(tmp16)(tmp16 * tmp18);
                    in_out_ptr0[static_cast<long>(x1 + (228L*x0))] = tmp19;
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_67 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(38L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp23 = in_ptr1[static_cast<long>(x0 + (38L*x1))];
                        auto tmp24 = in_ptr2[static_cast<long>(x0)];
                        auto tmp0 = c10::convert<long>(x0);
                        auto tmp1 = static_cast<long>(27);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr0[static_cast<long>(x0 + (38L*x1))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = tmp2 ? tmp5 : tmp6;
                        auto tmp8 = tmp0 < tmp1;
                        auto tmp9 = [&]
                        {
                            auto tmp10 = in_ptr0[static_cast<long>(x0 + (38L*x1))];
                            return tmp10;
                        }
                        ;
                        auto tmp11 = tmp8 ? tmp9() : static_cast<decltype(tmp9())>(0.0);
                        auto tmp12 = tmp8 ? tmp11 : tmp6;
                        auto tmp13 = decltype(tmp7)(tmp7 + tmp12);
                        auto tmp14 = [&]
                        {
                            auto tmp15 = in_ptr0[static_cast<long>(x0 + (38L*x1))];
                            return tmp15;
                        }
                        ;
                        auto tmp16 = tmp2 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                        auto tmp17 = tmp2 ? tmp16 : tmp6;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = in_ptr0[static_cast<long>(x0 + (38L*x1))];
                            return tmp19;
                        }
                        ;
                        auto tmp20 = tmp8 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                        auto tmp21 = tmp8 ? tmp20 : tmp6;
                        auto tmp22 = decltype(tmp17)(tmp17 + tmp21);
                        auto tmp25 = decltype(tmp23)(tmp23 - tmp24);
                        auto tmp26 = decltype(tmp22)(tmp22 * tmp25);
                        tmp_acc0 = tmp_acc0 + tmp13;
                        tmp_acc1 = tmp_acc1 + tmp26;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(32L); x0<static_cast<long>(38L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    out_ptr2[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(38L); x2+=static_cast<long>(1L))
                    {
                        auto tmp14 = in_ptr1[static_cast<long>(x2 + (38L*x1) + (119168L*x0))];
                        auto tmp15 = in_ptr2[static_cast<long>(x2)];
                        auto tmp17 = out_ptr1[static_cast<long>(x2)];
                        auto tmp20 = in_ptr3[static_cast<long>(x2)];
                        auto tmp25 = out_ptr0[static_cast<long>(x2)];
                        auto tmp28 = in_ptr4[static_cast<long>(x2)];
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(27);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_ptr0[static_cast<long>(x2 + (38L*x1) + (119168L*x0))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = tmp2 ? tmp5 : tmp6;
                        auto tmp8 = tmp0 < tmp1;
                        auto tmp9 = [&]
                        {
                            auto tmp10 = in_ptr0[static_cast<long>(x2 + (38L*x1) + (119168L*x0))];
                            return tmp10;
                        }
                        ;
                        auto tmp11 = tmp8 ? tmp9() : static_cast<decltype(tmp9())>(0.0);
                        auto tmp12 = tmp8 ? tmp11 : tmp6;
                        auto tmp13 = decltype(tmp7)(tmp7 + tmp12);
                        auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                        auto tmp18 = static_cast<float>(3.985969387755102e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                        auto tmp21 = decltype(tmp20)(tmp20 * tmp20);
                        auto tmp22 = decltype(tmp19)(tmp19 * tmp21);
                        auto tmp23 = decltype(tmp16)(tmp16 * tmp22);
                        auto tmp24 = decltype(tmp13)(tmp13 - tmp23);
                        auto tmp26 = decltype(tmp25)(tmp25 * tmp18);
                        auto tmp27 = decltype(tmp24)(tmp24 - tmp26);
                        auto tmp29 = decltype(tmp20)(tmp20 * tmp28);
                        auto tmp30 = decltype(tmp27)(tmp27 * tmp29);
                        out_ptr3[static_cast<long>(x1 + (3136L*x2) + (119168L*x0))] = tmp30;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (162L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (162L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (162L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp7 = tmp5 - tmp6;
                        auto tmp8 = tmp4 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(160L); x0<static_cast<long>(162L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (162L*x1))];
                        auto tmp1 = in_ptr1[static_cast<long>(x0 + (162L*x1))];
                        auto tmp4 = in_ptr2[static_cast<long>(x0 + (162L*x1))];
                        auto tmp5 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = tmp0 ? tmp2 : tmp1;
                        auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                        auto tmp7 = decltype(tmp3)(tmp3 * tmp6);
                        tmp_acc0 = tmp_acc0 + tmp3;
                        tmp_acc1 = tmp_acc1 + tmp7;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(160L); x0<static_cast<long>(162L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    out_ptr2[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (162L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (162L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (162L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    auto tmp7 = tmp5 - tmp6;
                    auto tmp9 = static_cast<float>(3.985969387755102e-05);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp4 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (162L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(160L); x1<static_cast<long>(162L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (162L*x0))];
                    auto tmp1 = in_out_ptr0[static_cast<long>(x1 + (162L*x0))];
                    auto tmp4 = in_ptr2[static_cast<long>(x1 + (162L*x0))];
                    auto tmp5 = in_ptr3[static_cast<long>(x1)];
                    auto tmp7 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr4[static_cast<long>(x1)];
                    auto tmp15 = out_ptr0[static_cast<long>(x1)];
                    auto tmp18 = in_ptr5[static_cast<long>(x1)];
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = tmp0 ? tmp2 : tmp1;
                    auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                    auto tmp8 = static_cast<float>(3.985969387755102e-05);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp11 = decltype(tmp10)(tmp10 * tmp10);
                    auto tmp12 = decltype(tmp9)(tmp9 * tmp11);
                    auto tmp13 = decltype(tmp6)(tmp6 * tmp12);
                    auto tmp14 = decltype(tmp3)(tmp3 - tmp13);
                    auto tmp16 = decltype(tmp15)(tmp15 * tmp8);
                    auto tmp17 = decltype(tmp14)(tmp14 - tmp16);
                    auto tmp19 = decltype(tmp10)(tmp10 * tmp18);
                    auto tmp20 = decltype(tmp17)(tmp17 * tmp19);
                    in_out_ptr0[static_cast<long>(x1 + (162L*x0))] = tmp20;
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (162L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (162L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (162L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(160L); x0<static_cast<long>(162L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (162L*x1))];
                        auto tmp1 = in_ptr1[static_cast<long>(x0 + (162L*x1))];
                        auto tmp3 = in_ptr2[static_cast<long>(x0 + (162L*x1))];
                        auto tmp4 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = decltype(tmp2)(tmp2 * tmp5);
                        tmp_acc0 = tmp_acc0 + tmp2;
                        tmp_acc1 = tmp_acc1 + tmp6;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(160L); x0<static_cast<long>(162L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    out_ptr2[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (162L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (162L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (162L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(3.985969387755102e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (162L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(160L); x1<static_cast<long>(162L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (162L*x0))];
                    auto tmp1 = in_ptr1[static_cast<long>(x1 + (162L*x0))];
                    auto tmp3 = in_ptr2[static_cast<long>(x1 + (162L*x0))];
                    auto tmp4 = in_ptr3[static_cast<long>(x1)];
                    auto tmp6 = out_ptr1[static_cast<long>(x1)];
                    auto tmp9 = in_ptr4[static_cast<long>(x1)];
                    auto tmp14 = out_ptr0[static_cast<long>(x1)];
                    auto tmp17 = in_ptr5[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                    auto tmp7 = static_cast<float>(3.985969387755102e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp10 = decltype(tmp9)(tmp9 * tmp9);
                    auto tmp11 = decltype(tmp8)(tmp8 * tmp10);
                    auto tmp12 = decltype(tmp5)(tmp5 * tmp11);
                    auto tmp13 = decltype(tmp2)(tmp2 - tmp12);
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp7);
                    auto tmp16 = decltype(tmp13)(tmp13 - tmp15);
                    auto tmp18 = decltype(tmp9)(tmp9 * tmp17);
                    auto tmp19 = decltype(tmp16)(tmp16 * tmp18);
                    in_out_ptr0[static_cast<long>(x1 + (162L*x0))] = tmp19;
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (38L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (27L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (27L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(24L); x0<static_cast<long>(27L); x0+=static_cast<long>(1L))
            {
                {
                    float tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (38L*x1))];
                        auto tmp1 = in_ptr1[static_cast<long>(x0 + (27L*x1))];
                        auto tmp3 = in_ptr2[static_cast<long>(x0 + (27L*x1))];
                        auto tmp4 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                        auto tmp6 = decltype(tmp2)(tmp2 * tmp5);
                        tmp_acc0 = tmp_acc0 + tmp2;
                        tmp_acc1 = tmp_acc1 + tmp6;
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(24L); x0<static_cast<long>(27L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    out_ptr2[static_cast<long>(x0)] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (38L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (27L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (27L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(3.985969387755102e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (27L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(24L); x1<static_cast<long>(27L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (38L*x0))];
                    auto tmp1 = in_out_ptr0[static_cast<long>(x1 + (27L*x0))];
                    auto tmp3 = in_ptr2[static_cast<long>(x1 + (27L*x0))];
                    auto tmp4 = in_ptr3[static_cast<long>(x1)];
                    auto tmp6 = out_ptr1[static_cast<long>(x1)];
                    auto tmp9 = in_ptr4[static_cast<long>(x1)];
                    auto tmp14 = out_ptr0[static_cast<long>(x1)];
                    auto tmp17 = in_ptr5[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    auto tmp5 = decltype(tmp3)(tmp3 - tmp4);
                    auto tmp7 = static_cast<float>(3.985969387755102e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp10 = decltype(tmp9)(tmp9 * tmp9);
                    auto tmp11 = decltype(tmp8)(tmp8 * tmp10);
                    auto tmp12 = decltype(tmp5)(tmp5 * tmp11);
                    auto tmp13 = decltype(tmp2)(tmp2 - tmp12);
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp7);
                    auto tmp16 = decltype(tmp13)(tmp13 - tmp15);
                    auto tmp18 = decltype(tmp9)(tmp9 * tmp17);
                    auto tmp19 = decltype(tmp16)(tmp16 * tmp18);
                    in_out_ptr0[static_cast<long>(x1 + (27L*x0))] = tmp19;
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp7 = tmp5 - tmp6;
                        auto tmp8 = tmp4 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    auto tmp7 = tmp5 - tmp6;
                    auto tmp9 = static_cast<float>(3.985969387755102e-05);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp4 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (96L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(9.964923469387754e-06);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (16L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (16L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(9.964923469387754e-06);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp8 * tmp8;
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp12 = tmp0 - tmp11;
                    auto tmp14 = tmp13 * tmp6;
                    auto tmp15 = tmp12 - tmp14;
                    auto tmp17 = tmp8 * tmp16;
                    auto tmp18 = tmp15 * tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                        auto tmp7 = tmp5 - tmp6;
                        auto tmp8 = tmp4 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = tmp_acc1_vec + tmp8;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = decltype(tmp3)::blendv(tmp1, tmp3, tmp0);
                    auto tmp7 = tmp5 - tmp6;
                    auto tmp9 = static_cast<float>(9.964923469387754e-06);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp4 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    tmp22.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_native_batch_norm_backward_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp7 = static_cast<float>(9.964923469387754e-06);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = tmp10 * tmp10;
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp14 = tmp2 - tmp13;
                    auto tmp16 = tmp15 * tmp8;
                    auto tmp17 = tmp14 - tmp16;
                    auto tmp19 = tmp10 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    tmp20.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_112, primals_114, primals_116, primals_117, primals_118, primals_119, primals_121, primals_123, primals_125, primals_126, primals_127, primals_128, primals_130, primals_132, primals_134, primals_135, primals_136, primals_137, primals_139, primals_141, primals_143, primals_144, primals_145, primals_146, primals_148, primals_150, primals_152, primals_153, primals_154, primals_155, primals_157, primals_159, primals_161, primals_162, primals_163, primals_164, primals_166, primals_168, primals_170, primals_171, primals_172, primals_173, primals_175, primals_177, primals_179, primals_180, primals_181, primals_182, primals_184, primals_186, primals_188, primals_189, primals_190, primals_191, primals_193, primals_195, primals_197, primals_198, primals_199, primals_200, primals_202, primals_204, primals_206, primals_207, primals_208, primals_209, primals_211, primals_213, primals_215, primals_216, primals_217, primals_218, primals_220, primals_222, primals_224, primals_225, primals_414, convolution, squeeze_1, mul_7, convolution_1, squeeze_4, clamp_max, convolution_2, squeeze_7, add_14, convolution_3, squeeze_10, mul_29, convolution_4, squeeze_13, clamp_max_1, convolution_5, squeeze_16, add_29, convolution_6, squeeze_19, mul_51, convolution_7, squeeze_22, clamp_max_2, convolution_8, squeeze_25, cat, convolution_9, squeeze_28, mul_73, convolution_10, squeeze_31, add_55, mean, convolution_11, relu, convolution_12, clamp_max_3, convolution_13, squeeze_37, add_65, convolution_14, squeeze_40, mul_103, convolution_15, squeeze_43, add_75, mean_1, convolution_16, relu_1, convolution_17, clamp_max_4, convolution_18, squeeze_49, cat_1, convolution_19, squeeze_52, mul_133, convolution_20, squeeze_55, add_96, mean_2, convolution_21, relu_2, convolution_22, clamp_max_5, convolution_23, squeeze_61, add_106, convolution_24, squeeze_64, mul_163, convolution_25, squeeze_67, add_116, mean_3, convolution_26, relu_3, convolution_27, clamp_max_6, convolution_28, squeeze_73, cat_2, convolution_29, squeeze_76, mul_193, convolution_30, squeeze_79, add_137, mean_4, convolution_31, relu_4, convolution_32, clamp_max_7, convolution_33, squeeze_85, cat_3, convolution_34, squeeze_88, mul_223, convolution_35, squeeze_91, add_158, mean_5, convolution_36, relu_5, convolution_37, clamp_max_8, convolution_38, squeeze_97, cat_4, convolution_39, squeeze_100, mul_253, convolution_40, squeeze_103, add_179, mean_6, convolution_41, relu_6, convolution_42, clamp_max_9, convolution_43, squeeze_109, cat_5, convolution_44, squeeze_112, mul_283, convolution_45, squeeze_115, add_200, mean_7, convolution_46, relu_7, convolution_47, clamp_max_10, convolution_48, squeeze_121, cat_6, convolution_49, squeeze_124, mul_313, convolution_50, squeeze_127, add_221, mean_8, convolution_51, relu_8, convolution_52, clamp_max_11, convolution_53, squeeze_133, add_231, convolution_54, squeeze_136, mul_343, convolution_55, squeeze_139, add_241, mean_9, convolution_56, relu_9, convolution_57, clamp_max_12, convolution_58, squeeze_145, cat_7, convolution_59, squeeze_148, mul_373, convolution_60, squeeze_151, add_262, mean_10, convolution_61, relu_10, convolution_62, clamp_max_13, convolution_63, squeeze_157, cat_8, convolution_64, squeeze_160, mul_403, convolution_65, squeeze_163, add_283, mean_11, convolution_66, relu_11, convolution_67, clamp_max_14, convolution_68, squeeze_169, cat_9, convolution_69, squeeze_172, mul_433, convolution_70, squeeze_175, add_304, mean_12, convolution_71, relu_12, convolution_72, clamp_max_15, convolution_73, squeeze_181, cat_10, convolution_74, squeeze_184, clone_17, permute_1, mul_465, unsqueeze_250, unsqueeze_262, unsqueeze_286, mul_508, unsqueeze_298, unsqueeze_310, unsqueeze_334, mul_551, unsqueeze_346, unsqueeze_358, unsqueeze_382, mul_594, unsqueeze_394, unsqueeze_406, unsqueeze_430, mul_637, unsqueeze_442, unsqueeze_454, unsqueeze_478, mul_680, unsqueeze_490, unsqueeze_502, unsqueeze_526, mul_723, unsqueeze_538, unsqueeze_550, unsqueeze_574, mul_766, unsqueeze_586, unsqueeze_598, unsqueeze_622, mul_809, unsqueeze_634, unsqueeze_646, unsqueeze_670, mul_852, unsqueeze_682, unsqueeze_694, unsqueeze_718, mul_895, unsqueeze_730, unsqueeze_742, unsqueeze_766, mul_938, unsqueeze_778, unsqueeze_790, unsqueeze_814, mul_981, unsqueeze_826, unsqueeze_838, unsqueeze_862, mul_1024, unsqueeze_874, unsqueeze_886, bitwise_or_13, unsqueeze_898, mul_1054, unsqueeze_910, unsqueeze_922, bitwise_or_14, unsqueeze_934, mul_1084, unsqueeze_946, unsqueeze_958, bitwise_or_15, unsqueeze_970, mul_1114, unsqueeze_982, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (32, ), (1, ))
    assert_size_stride(primals_3, (32, ), (1, ))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_7, (96, ), (1, ))
    assert_size_stride(primals_9, (96, ), (1, ))
    assert_size_stride(primals_11, (27, ), (1, ))
    assert_size_stride(primals_13, (162, ), (1, ))
    assert_size_stride(primals_15, (162, ), (1, ))
    assert_size_stride(primals_17, (38, ), (1, ))
    assert_size_stride(primals_19, (228, ), (1, ))
    assert_size_stride(primals_21, (228, ), (1, ))
    assert_size_stride(primals_23, (50, ), (1, ))
    assert_size_stride(primals_25, (300, ), (1, ))
    assert_size_stride(primals_27, (300, ), (1, ))
    assert_size_stride(primals_29, (61, ), (1, ))
    assert_size_stride(primals_31, (366, ), (1, ))
    assert_size_stride(primals_33, (366, ), (1, ))
    assert_size_stride(primals_35, (72, ), (1, ))
    assert_size_stride(primals_37, (432, ), (1, ))
    assert_size_stride(primals_39, (432, ), (1, ))
    assert_size_stride(primals_41, (84, ), (1, ))
    assert_size_stride(primals_43, (504, ), (1, ))
    assert_size_stride(primals_45, (504, ), (1, ))
    assert_size_stride(primals_47, (95, ), (1, ))
    assert_size_stride(primals_49, (570, ), (1, ))
    assert_size_stride(primals_51, (570, ), (1, ))
    assert_size_stride(primals_53, (106, ), (1, ))
    assert_size_stride(primals_55, (636, ), (1, ))
    assert_size_stride(primals_57, (636, ), (1, ))
    assert_size_stride(primals_59, (117, ), (1, ))
    assert_size_stride(primals_61, (702, ), (1, ))
    assert_size_stride(primals_63, (702, ), (1, ))
    assert_size_stride(primals_65, (128, ), (1, ))
    assert_size_stride(primals_67, (768, ), (1, ))
    assert_size_stride(primals_69, (768, ), (1, ))
    assert_size_stride(primals_71, (140, ), (1, ))
    assert_size_stride(primals_73, (840, ), (1, ))
    assert_size_stride(primals_75, (840, ), (1, ))
    assert_size_stride(primals_77, (151, ), (1, ))
    assert_size_stride(primals_79, (906, ), (1, ))
    assert_size_stride(primals_81, (906, ), (1, ))
    assert_size_stride(primals_83, (162, ), (1, ))
    assert_size_stride(primals_85, (972, ), (1, ))
    assert_size_stride(primals_87, (972, ), (1, ))
    assert_size_stride(primals_89, (174, ), (1, ))
    assert_size_stride(primals_91, (1044, ), (1, ))
    assert_size_stride(primals_93, (1044, ), (1, ))
    assert_size_stride(primals_95, (185, ), (1, ))
    assert_size_stride(primals_97, (1280, ), (1, ))
    assert_size_stride(primals_99, (32, 3, 3, 3), (27, 1, 9, 3))
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
    assert_size_stride(primals_112, (19, ), (1, ))
    assert_size_stride(primals_114, (228, 19, 1, 1), (19, 1, 1, 1))
    assert_size_stride(primals_116, (50, 228, 1, 1), (228, 1, 1, 1))
    assert_size_stride(primals_117, (300, 50, 1, 1), (50, 1, 1, 1))
    assert_size_stride(primals_118, (300, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_119, (25, 300, 1, 1), (300, 1, 1, 1))
    assert_size_stride(primals_121, (25, ), (1, ))
    assert_size_stride(primals_123, (300, 25, 1, 1), (25, 1, 1, 1))
    assert_size_stride(primals_125, (61, 300, 1, 1), (300, 1, 1, 1))
    assert_size_stride(primals_126, (366, 61, 1, 1), (61, 1, 1, 1))
    assert_size_stride(primals_127, (366, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_128, (30, 366, 1, 1), (366, 1, 1, 1))
    assert_size_stride(primals_130, (30, ), (1, ))
    assert_size_stride(primals_132, (366, 30, 1, 1), (30, 1, 1, 1))
    assert_size_stride(primals_134, (72, 366, 1, 1), (366, 1, 1, 1))
    assert_size_stride(primals_135, (432, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_136, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_137, (36, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(primals_139, (36, ), (1, ))
    assert_size_stride(primals_141, (432, 36, 1, 1), (36, 1, 1, 1))
    assert_size_stride(primals_143, (84, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(primals_144, (504, 84, 1, 1), (84, 1, 1, 1))
    assert_size_stride(primals_145, (504, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_146, (42, 504, 1, 1), (504, 1, 1, 1))
    assert_size_stride(primals_148, (42, ), (1, ))
    assert_size_stride(primals_150, (504, 42, 1, 1), (42, 1, 1, 1))
    assert_size_stride(primals_152, (95, 504, 1, 1), (504, 1, 1, 1))
    assert_size_stride(primals_153, (570, 95, 1, 1), (95, 1, 1, 1))
    assert_size_stride(primals_154, (570, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_155, (47, 570, 1, 1), (570, 1, 1, 1))
    assert_size_stride(primals_157, (47, ), (1, ))
    assert_size_stride(primals_159, (570, 47, 1, 1), (47, 1, 1, 1))
    assert_size_stride(primals_161, (106, 570, 1, 1), (570, 1, 1, 1))
    assert_size_stride(primals_162, (636, 106, 1, 1), (106, 1, 1, 1))
    assert_size_stride(primals_163, (636, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_164, (53, 636, 1, 1), (636, 1, 1, 1))
    assert_size_stride(primals_166, (53, ), (1, ))
    assert_size_stride(primals_168, (636, 53, 1, 1), (53, 1, 1, 1))
    assert_size_stride(primals_170, (117, 636, 1, 1), (636, 1, 1, 1))
    assert_size_stride(primals_171, (702, 117, 1, 1), (117, 1, 1, 1))
    assert_size_stride(primals_172, (702, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_173, (58, 702, 1, 1), (702, 1, 1, 1))
    assert_size_stride(primals_175, (58, ), (1, ))
    assert_size_stride(primals_177, (702, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(primals_179, (128, 702, 1, 1), (702, 1, 1, 1))
    assert_size_stride(primals_180, (768, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_181, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_182, (64, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_184, (64, ), (1, ))
    assert_size_stride(primals_186, (768, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_188, (140, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_189, (840, 140, 1, 1), (140, 1, 1, 1))
    assert_size_stride(primals_190, (840, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_191, (70, 840, 1, 1), (840, 1, 1, 1))
    assert_size_stride(primals_193, (70, ), (1, ))
    assert_size_stride(primals_195, (840, 70, 1, 1), (70, 1, 1, 1))
    assert_size_stride(primals_197, (151, 840, 1, 1), (840, 1, 1, 1))
    assert_size_stride(primals_198, (906, 151, 1, 1), (151, 1, 1, 1))
    assert_size_stride(primals_199, (906, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_200, (75, 906, 1, 1), (906, 1, 1, 1))
    assert_size_stride(primals_202, (75, ), (1, ))
    assert_size_stride(primals_204, (906, 75, 1, 1), (75, 1, 1, 1))
    assert_size_stride(primals_206, (162, 906, 1, 1), (906, 1, 1, 1))
    assert_size_stride(primals_207, (972, 162, 1, 1), (162, 1, 1, 1))
    assert_size_stride(primals_208, (972, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_209, (81, 972, 1, 1), (972, 1, 1, 1))
    assert_size_stride(primals_211, (81, ), (1, ))
    assert_size_stride(primals_213, (972, 81, 1, 1), (81, 1, 1, 1))
    assert_size_stride(primals_215, (174, 972, 1, 1), (972, 1, 1, 1))
    assert_size_stride(primals_216, (1044, 174, 1, 1), (174, 1, 1, 1))
    assert_size_stride(primals_217, (1044, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_218, (87, 1044, 1, 1), (1044, 1, 1, 1))
    assert_size_stride(primals_220, (87, ), (1, ))
    assert_size_stride(primals_222, (1044, 87, 1, 1), (87, 1, 1, 1))
    assert_size_stride(primals_224, (185, 1044, 1, 1), (1044, 1, 1, 1))
    assert_size_stride(primals_225, (1280, 185, 1, 1), (185, 1, 1, 1))
    assert_size_stride(primals_414, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(squeeze_1, (32, ), (1, ))
    assert_size_stride(mul_7, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(convolution_1, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(squeeze_4, (32, ), (1, ))
    assert_size_stride(clamp_max, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(convolution_2, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(squeeze_7, (16, ), (1, ))
    assert_size_stride(add_14, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(convolution_3, (8, 96, 112, 112), (1204224, 1, 10752, 96))
    assert_size_stride(squeeze_10, (96, ), (1, ))
    assert_size_stride(mul_29, (8, 96, 112, 112), (1204224, 1, 10752, 96))
    assert_size_stride(convolution_4, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(squeeze_13, (96, ), (1, ))
    assert_size_stride(clamp_max_1, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(convolution_5, (8, 27, 56, 56), (84672, 1, 1512, 27))
    assert_size_stride(squeeze_16, (27, ), (1, ))
    assert_size_stride(add_29, (8, 27, 56, 56), (84672, 1, 1512, 27))
    assert_size_stride(convolution_6, (8, 162, 56, 56), (508032, 1, 9072, 162))
    assert_size_stride(squeeze_19, (162, ), (1, ))
    assert_size_stride(mul_51, (8, 162, 56, 56), (508032, 1, 9072, 162))
    assert_size_stride(convolution_7, (8, 162, 56, 56), (508032, 1, 9072, 162))
    assert_size_stride(squeeze_22, (162, ), (1, ))
    assert_size_stride(clamp_max_2, (8, 162, 56, 56), (508032, 1, 9072, 162))
    assert_size_stride(convolution_8, (8, 38, 56, 56), (119168, 1, 2128, 38))
    assert_size_stride(squeeze_25, (38, ), (1, ))
    assert_size_stride(cat, (8, 38, 56, 56), (119168, 1, 2128, 38))
    assert_size_stride(convolution_9, (8, 228, 56, 56), (715008, 1, 12768, 228))
    assert_size_stride(squeeze_28, (228, ), (1, ))
    assert_size_stride(mul_73, (8, 228, 56, 56), (715008, 1, 12768, 228))
    assert_size_stride(convolution_10, (8, 228, 28, 28), (178752, 1, 6384, 228))
    assert_size_stride(squeeze_31, (228, ), (1, ))
    assert_size_stride(add_55, (8, 228, 28, 28), (178752, 1, 6384, 228))
    assert_size_stride(mean, (8, 228, 1, 1), (228, 1, 228, 228))
    assert_size_stride(convolution_11, (8, 19, 1, 1), (19, 1, 19, 19))
    assert_size_stride(relu, (8, 19, 1, 1), (19, 1, 19, 19))
    assert_size_stride(convolution_12, (8, 228, 1, 1), (228, 1, 228, 228))
    assert_size_stride(clamp_max_3, (8, 228, 28, 28), (178752, 1, 6384, 228))
    assert_size_stride(convolution_13, (8, 50, 28, 28), (39200, 1, 1400, 50))
    assert_size_stride(squeeze_37, (50, ), (1, ))
    assert_size_stride(add_65, (8, 50, 28, 28), (39200, 1, 1400, 50))
    assert_size_stride(convolution_14, (8, 300, 28, 28), (235200, 1, 8400, 300))
    assert_size_stride(squeeze_40, (300, ), (1, ))
    assert_size_stride(mul_103, (8, 300, 28, 28), (235200, 1, 8400, 300))
    assert_size_stride(convolution_15, (8, 300, 28, 28), (235200, 1, 8400, 300))
    assert_size_stride(squeeze_43, (300, ), (1, ))
    assert_size_stride(add_75, (8, 300, 28, 28), (235200, 1, 8400, 300))
    assert_size_stride(mean_1, (8, 300, 1, 1), (300, 1, 300, 300))
    assert_size_stride(convolution_16, (8, 25, 1, 1), (25, 1, 25, 25))
    assert_size_stride(relu_1, (8, 25, 1, 1), (25, 1, 25, 25))
    assert_size_stride(convolution_17, (8, 300, 1, 1), (300, 1, 300, 300))
    assert_size_stride(clamp_max_4, (8, 300, 28, 28), (235200, 1, 8400, 300))
    assert_size_stride(convolution_18, (8, 61, 28, 28), (47824, 1, 1708, 61))
    assert_size_stride(squeeze_49, (61, ), (1, ))
    assert_size_stride(cat_1, (8, 61, 28, 28), (47824, 1, 1708, 61))
    assert_size_stride(convolution_19, (8, 366, 28, 28), (286944, 1, 10248, 366))
    assert_size_stride(squeeze_52, (366, ), (1, ))
    assert_size_stride(mul_133, (8, 366, 28, 28), (286944, 1, 10248, 366))
    assert_size_stride(convolution_20, (8, 366, 14, 14), (71736, 1, 5124, 366))
    assert_size_stride(squeeze_55, (366, ), (1, ))
    assert_size_stride(add_96, (8, 366, 14, 14), (71736, 1, 5124, 366))
    assert_size_stride(mean_2, (8, 366, 1, 1), (366, 1, 366, 366))
    assert_size_stride(convolution_21, (8, 30, 1, 1), (30, 1, 30, 30))
    assert_size_stride(relu_2, (8, 30, 1, 1), (30, 1, 30, 30))
    assert_size_stride(convolution_22, (8, 366, 1, 1), (366, 1, 366, 366))
    assert_size_stride(clamp_max_5, (8, 366, 14, 14), (71736, 1, 5124, 366))
    assert_size_stride(convolution_23, (8, 72, 14, 14), (14112, 1, 1008, 72))
    assert_size_stride(squeeze_61, (72, ), (1, ))
    assert_size_stride(add_106, (8, 72, 14, 14), (14112, 1, 1008, 72))
    assert_size_stride(convolution_24, (8, 432, 14, 14), (84672, 1, 6048, 432))
    assert_size_stride(squeeze_64, (432, ), (1, ))
    assert_size_stride(mul_163, (8, 432, 14, 14), (84672, 1, 6048, 432))
    assert_size_stride(convolution_25, (8, 432, 14, 14), (84672, 1, 6048, 432))
    assert_size_stride(squeeze_67, (432, ), (1, ))
    assert_size_stride(add_116, (8, 432, 14, 14), (84672, 1, 6048, 432))
    assert_size_stride(mean_3, (8, 432, 1, 1), (432, 1, 432, 432))
    assert_size_stride(convolution_26, (8, 36, 1, 1), (36, 1, 36, 36))
    assert_size_stride(relu_3, (8, 36, 1, 1), (36, 1, 36, 36))
    assert_size_stride(convolution_27, (8, 432, 1, 1), (432, 1, 432, 432))
    assert_size_stride(clamp_max_6, (8, 432, 14, 14), (84672, 1, 6048, 432))
    assert_size_stride(convolution_28, (8, 84, 14, 14), (16464, 1, 1176, 84))
    assert_size_stride(squeeze_73, (84, ), (1, ))
    assert_size_stride(cat_2, (8, 84, 14, 14), (16464, 1, 1176, 84))
    assert_size_stride(convolution_29, (8, 504, 14, 14), (98784, 1, 7056, 504))
    assert_size_stride(squeeze_76, (504, ), (1, ))
    assert_size_stride(mul_193, (8, 504, 14, 14), (98784, 1, 7056, 504))
    assert_size_stride(convolution_30, (8, 504, 14, 14), (98784, 1, 7056, 504))
    assert_size_stride(squeeze_79, (504, ), (1, ))
    assert_size_stride(add_137, (8, 504, 14, 14), (98784, 1, 7056, 504))
    assert_size_stride(mean_4, (8, 504, 1, 1), (504, 1, 504, 504))
    assert_size_stride(convolution_31, (8, 42, 1, 1), (42, 1, 42, 42))
    assert_size_stride(relu_4, (8, 42, 1, 1), (42, 1, 42, 42))
    assert_size_stride(convolution_32, (8, 504, 1, 1), (504, 1, 504, 504))
    assert_size_stride(clamp_max_7, (8, 504, 14, 14), (98784, 1, 7056, 504))
    assert_size_stride(convolution_33, (8, 95, 14, 14), (18620, 1, 1330, 95))
    assert_size_stride(squeeze_85, (95, ), (1, ))
    assert_size_stride(cat_3, (8, 95, 14, 14), (18620, 1, 1330, 95))
    assert_size_stride(convolution_34, (8, 570, 14, 14), (111720, 1, 7980, 570))
    assert_size_stride(squeeze_88, (570, ), (1, ))
    assert_size_stride(mul_223, (8, 570, 14, 14), (111720, 1, 7980, 570))
    assert_size_stride(convolution_35, (8, 570, 14, 14), (111720, 1, 7980, 570))
    assert_size_stride(squeeze_91, (570, ), (1, ))
    assert_size_stride(add_158, (8, 570, 14, 14), (111720, 1, 7980, 570))
    assert_size_stride(mean_5, (8, 570, 1, 1), (570, 1, 570, 570))
    assert_size_stride(convolution_36, (8, 47, 1, 1), (47, 1, 47, 47))
    assert_size_stride(relu_5, (8, 47, 1, 1), (47, 1, 47, 47))
    assert_size_stride(convolution_37, (8, 570, 1, 1), (570, 1, 570, 570))
    assert_size_stride(clamp_max_8, (8, 570, 14, 14), (111720, 1, 7980, 570))
    assert_size_stride(convolution_38, (8, 106, 14, 14), (20776, 1, 1484, 106))
    assert_size_stride(squeeze_97, (106, ), (1, ))
    assert_size_stride(cat_4, (8, 106, 14, 14), (20776, 1, 1484, 106))
    assert_size_stride(convolution_39, (8, 636, 14, 14), (124656, 1, 8904, 636))
    assert_size_stride(squeeze_100, (636, ), (1, ))
    assert_size_stride(mul_253, (8, 636, 14, 14), (124656, 1, 8904, 636))
    assert_size_stride(convolution_40, (8, 636, 14, 14), (124656, 1, 8904, 636))
    assert_size_stride(squeeze_103, (636, ), (1, ))
    assert_size_stride(add_179, (8, 636, 14, 14), (124656, 1, 8904, 636))
    assert_size_stride(mean_6, (8, 636, 1, 1), (636, 1, 636, 636))
    assert_size_stride(convolution_41, (8, 53, 1, 1), (53, 1, 53, 53))
    assert_size_stride(relu_6, (8, 53, 1, 1), (53, 1, 53, 53))
    assert_size_stride(convolution_42, (8, 636, 1, 1), (636, 1, 636, 636))
    assert_size_stride(clamp_max_9, (8, 636, 14, 14), (124656, 1, 8904, 636))
    assert_size_stride(convolution_43, (8, 117, 14, 14), (22932, 1, 1638, 117))
    assert_size_stride(squeeze_109, (117, ), (1, ))
    assert_size_stride(cat_5, (8, 117, 14, 14), (22932, 1, 1638, 117))
    assert_size_stride(convolution_44, (8, 702, 14, 14), (137592, 1, 9828, 702))
    assert_size_stride(squeeze_112, (702, ), (1, ))
    assert_size_stride(mul_283, (8, 702, 14, 14), (137592, 1, 9828, 702))
    assert_size_stride(convolution_45, (8, 702, 14, 14), (137592, 1, 9828, 702))
    assert_size_stride(squeeze_115, (702, ), (1, ))
    assert_size_stride(add_200, (8, 702, 14, 14), (137592, 1, 9828, 702))
    assert_size_stride(mean_7, (8, 702, 1, 1), (702, 1, 702, 702))
    assert_size_stride(convolution_46, (8, 58, 1, 1), (58, 1, 58, 58))
    assert_size_stride(relu_7, (8, 58, 1, 1), (58, 1, 58, 58))
    assert_size_stride(convolution_47, (8, 702, 1, 1), (702, 1, 702, 702))
    assert_size_stride(clamp_max_10, (8, 702, 14, 14), (137592, 1, 9828, 702))
    assert_size_stride(convolution_48, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_121, (128, ), (1, ))
    assert_size_stride(cat_6, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(convolution_49, (8, 768, 14, 14), (150528, 1, 10752, 768))
    assert_size_stride(squeeze_124, (768, ), (1, ))
    assert_size_stride(mul_313, (8, 768, 14, 14), (150528, 1, 10752, 768))
    assert_size_stride(convolution_50, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(squeeze_127, (768, ), (1, ))
    assert_size_stride(add_221, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(mean_8, (8, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(convolution_51, (8, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(relu_8, (8, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(convolution_52, (8, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(clamp_max_11, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(convolution_53, (8, 140, 7, 7), (6860, 1, 980, 140))
    assert_size_stride(squeeze_133, (140, ), (1, ))
    assert_size_stride(add_231, (8, 140, 7, 7), (6860, 1, 980, 140))
    assert_size_stride(convolution_54, (8, 840, 7, 7), (41160, 1, 5880, 840))
    assert_size_stride(squeeze_136, (840, ), (1, ))
    assert_size_stride(mul_343, (8, 840, 7, 7), (41160, 1, 5880, 840))
    assert_size_stride(convolution_55, (8, 840, 7, 7), (41160, 1, 5880, 840))
    assert_size_stride(squeeze_139, (840, ), (1, ))
    assert_size_stride(add_241, (8, 840, 7, 7), (41160, 1, 5880, 840))
    assert_size_stride(mean_9, (8, 840, 1, 1), (840, 1, 840, 840))
    assert_size_stride(convolution_56, (8, 70, 1, 1), (70, 1, 70, 70))
    assert_size_stride(relu_9, (8, 70, 1, 1), (70, 1, 70, 70))
    assert_size_stride(convolution_57, (8, 840, 1, 1), (840, 1, 840, 840))
    assert_size_stride(clamp_max_12, (8, 840, 7, 7), (41160, 1, 5880, 840))
    assert_size_stride(convolution_58, (8, 151, 7, 7), (7399, 1, 1057, 151))
    assert_size_stride(squeeze_145, (151, ), (1, ))
    assert_size_stride(cat_7, (8, 151, 7, 7), (7399, 1, 1057, 151))
    assert_size_stride(convolution_59, (8, 906, 7, 7), (44394, 1, 6342, 906))
    assert_size_stride(squeeze_148, (906, ), (1, ))
    assert_size_stride(mul_373, (8, 906, 7, 7), (44394, 1, 6342, 906))
    assert_size_stride(convolution_60, (8, 906, 7, 7), (44394, 1, 6342, 906))
    assert_size_stride(squeeze_151, (906, ), (1, ))
    assert_size_stride(add_262, (8, 906, 7, 7), (44394, 1, 6342, 906))
    assert_size_stride(mean_10, (8, 906, 1, 1), (906, 1, 906, 906))
    assert_size_stride(convolution_61, (8, 75, 1, 1), (75, 1, 75, 75))
    assert_size_stride(relu_10, (8, 75, 1, 1), (75, 1, 75, 75))
    assert_size_stride(convolution_62, (8, 906, 1, 1), (906, 1, 906, 906))
    assert_size_stride(clamp_max_13, (8, 906, 7, 7), (44394, 1, 6342, 906))
    assert_size_stride(convolution_63, (8, 162, 7, 7), (7938, 1, 1134, 162))
    assert_size_stride(squeeze_157, (162, ), (1, ))
    assert_size_stride(cat_8, (8, 162, 7, 7), (7938, 1, 1134, 162))
    assert_size_stride(convolution_64, (8, 972, 7, 7), (47628, 1, 6804, 972))
    assert_size_stride(squeeze_160, (972, ), (1, ))
    assert_size_stride(mul_403, (8, 972, 7, 7), (47628, 1, 6804, 972))
    assert_size_stride(convolution_65, (8, 972, 7, 7), (47628, 1, 6804, 972))
    assert_size_stride(squeeze_163, (972, ), (1, ))
    assert_size_stride(add_283, (8, 972, 7, 7), (47628, 1, 6804, 972))
    assert_size_stride(mean_11, (8, 972, 1, 1), (972, 1, 972, 972))
    assert_size_stride(convolution_66, (8, 81, 1, 1), (81, 1, 81, 81))
    assert_size_stride(relu_11, (8, 81, 1, 1), (81, 1, 81, 81))
    assert_size_stride(convolution_67, (8, 972, 1, 1), (972, 1, 972, 972))
    assert_size_stride(clamp_max_14, (8, 972, 7, 7), (47628, 1, 6804, 972))
    assert_size_stride(convolution_68, (8, 174, 7, 7), (8526, 1, 1218, 174))
    assert_size_stride(squeeze_169, (174, ), (1, ))
    assert_size_stride(cat_9, (8, 174, 7, 7), (8526, 1, 1218, 174))
    assert_size_stride(convolution_69, (8, 1044, 7, 7), (51156, 1, 7308, 1044))
    assert_size_stride(squeeze_172, (1044, ), (1, ))
    assert_size_stride(mul_433, (8, 1044, 7, 7), (51156, 1, 7308, 1044))
    assert_size_stride(convolution_70, (8, 1044, 7, 7), (51156, 1, 7308, 1044))
    assert_size_stride(squeeze_175, (1044, ), (1, ))
    assert_size_stride(add_304, (8, 1044, 7, 7), (51156, 1, 7308, 1044))
    assert_size_stride(mean_12, (8, 1044, 1, 1), (1044, 1, 1044, 1044))
    assert_size_stride(convolution_71, (8, 87, 1, 1), (87, 1, 87, 87))
    assert_size_stride(relu_12, (8, 87, 1, 1), (87, 1, 87, 87))
    assert_size_stride(convolution_72, (8, 1044, 1, 1), (1044, 1, 1044, 1044))
    assert_size_stride(clamp_max_15, (8, 1044, 7, 7), (51156, 1, 7308, 1044))
    assert_size_stride(convolution_73, (8, 185, 7, 7), (9065, 1, 1295, 185))
    assert_size_stride(squeeze_181, (185, ), (1, ))
    assert_size_stride(cat_10, (8, 185, 7, 7), (9065, 1, 1295, 185))
    assert_size_stride(convolution_74, (8, 1280, 7, 7), (62720, 1, 8960, 1280))
    assert_size_stride(squeeze_184, (1280, ), (1, ))
    assert_size_stride(clone_17, (8, 1280), (1280, 1))
    assert_size_stride(permute_1, (1000, 1280), (1280, 1))
    assert_size_stride(mul_465, (8, 1280, 7, 7), (62720, 1, 8960, 1280))
    assert_size_stride(unsqueeze_250, (1, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(unsqueeze_262, (1, 185, 1, 1), (185, 1, 1, 1))
    assert_size_stride(unsqueeze_286, (1, 1044, 1, 1), (1044, 1, 1, 1))
    assert_size_stride(mul_508, (8, 1044, 7, 7), (51156, 1, 7308, 1044))
    assert_size_stride(unsqueeze_298, (1, 1044, 1, 1), (1044, 1, 1, 1))
    assert_size_stride(unsqueeze_310, (1, 174, 1, 1), (174, 1, 1, 1))
    assert_size_stride(unsqueeze_334, (1, 972, 1, 1), (972, 1, 1, 1))
    assert_size_stride(mul_551, (8, 972, 7, 7), (47628, 1, 6804, 972))
    assert_size_stride(unsqueeze_346, (1, 972, 1, 1), (972, 1, 1, 1))
    assert_size_stride(unsqueeze_358, (1, 162, 1, 1), (162, 1, 1, 1))
    assert_size_stride(unsqueeze_382, (1, 906, 1, 1), (906, 1, 1, 1))
    assert_size_stride(mul_594, (8, 906, 7, 7), (44394, 1, 6342, 906))
    assert_size_stride(unsqueeze_394, (1, 906, 1, 1), (906, 1, 1, 1))
    assert_size_stride(unsqueeze_406, (1, 151, 1, 1), (151, 1, 1, 1))
    assert_size_stride(unsqueeze_430, (1, 840, 1, 1), (840, 1, 1, 1))
    assert_size_stride(mul_637, (8, 840, 7, 7), (41160, 1, 5880, 840))
    assert_size_stride(unsqueeze_442, (1, 840, 1, 1), (840, 1, 1, 1))
    assert_size_stride(unsqueeze_454, (1, 140, 1, 1), (140, 1, 1, 1))
    assert_size_stride(unsqueeze_478, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(mul_680, (8, 768, 14, 14), (150528, 1, 10752, 768))
    assert_size_stride(unsqueeze_490, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_502, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_526, (1, 702, 1, 1), (702, 1, 1, 1))
    assert_size_stride(mul_723, (8, 702, 14, 14), (137592, 1, 9828, 702))
    assert_size_stride(unsqueeze_538, (1, 702, 1, 1), (702, 1, 1, 1))
    assert_size_stride(unsqueeze_550, (1, 117, 1, 1), (117, 1, 1, 1))
    assert_size_stride(unsqueeze_574, (1, 636, 1, 1), (636, 1, 1, 1))
    assert_size_stride(mul_766, (8, 636, 14, 14), (124656, 1, 8904, 636))
    assert_size_stride(unsqueeze_586, (1, 636, 1, 1), (636, 1, 1, 1))
    assert_size_stride(unsqueeze_598, (1, 106, 1, 1), (106, 1, 1, 1))
    assert_size_stride(unsqueeze_622, (1, 570, 1, 1), (570, 1, 1, 1))
    assert_size_stride(mul_809, (8, 570, 14, 14), (111720, 1, 7980, 570))
    assert_size_stride(unsqueeze_634, (1, 570, 1, 1), (570, 1, 1, 1))
    assert_size_stride(unsqueeze_646, (1, 95, 1, 1), (95, 1, 1, 1))
    assert_size_stride(unsqueeze_670, (1, 504, 1, 1), (504, 1, 1, 1))
    assert_size_stride(mul_852, (8, 504, 14, 14), (98784, 1, 7056, 504))
    assert_size_stride(unsqueeze_682, (1, 504, 1, 1), (504, 1, 1, 1))
    assert_size_stride(unsqueeze_694, (1, 84, 1, 1), (84, 1, 1, 1))
    assert_size_stride(unsqueeze_718, (1, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(mul_895, (8, 432, 14, 14), (84672, 1, 6048, 432))
    assert_size_stride(unsqueeze_730, (1, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(unsqueeze_742, (1, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(unsqueeze_766, (1, 366, 1, 1), (366, 1, 1, 1))
    assert_size_stride(mul_938, (8, 366, 28, 28), (286944, 1, 10248, 366))
    assert_size_stride(unsqueeze_778, (1, 366, 1, 1), (366, 1, 1, 1))
    assert_size_stride(unsqueeze_790, (1, 61, 1, 1), (61, 1, 1, 1))
    assert_size_stride(unsqueeze_814, (1, 300, 1, 1), (300, 1, 1, 1))
    assert_size_stride(mul_981, (8, 300, 28, 28), (235200, 1, 8400, 300))
    assert_size_stride(unsqueeze_826, (1, 300, 1, 1), (300, 1, 1, 1))
    assert_size_stride(unsqueeze_838, (1, 50, 1, 1), (50, 1, 1, 1))
    assert_size_stride(unsqueeze_862, (1, 228, 1, 1), (228, 1, 1, 1))
    assert_size_stride(mul_1024, (8, 228, 56, 56), (715008, 1, 12768, 228))
    assert_size_stride(unsqueeze_874, (1, 228, 1, 1), (228, 1, 1, 1))
    assert_size_stride(unsqueeze_886, (1, 38, 1, 1), (38, 1, 1, 1))
    assert_size_stride(bitwise_or_13, (8, 162, 56, 56), (508032, 1, 9072, 162))
    assert_size_stride(unsqueeze_898, (1, 162, 1, 1), (162, 1, 1, 1))
    assert_size_stride(mul_1054, (8, 162, 56, 56), (508032, 1, 9072, 162))
    assert_size_stride(unsqueeze_910, (1, 162, 1, 1), (162, 1, 1, 1))
    assert_size_stride(unsqueeze_922, (1, 27, 1, 1), (27, 1, 1, 1))
    assert_size_stride(bitwise_or_14, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(unsqueeze_934, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(mul_1084, (8, 96, 112, 112), (1204224, 1, 10752, 96))
    assert_size_stride(unsqueeze_946, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_958, (1, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(bitwise_or_15, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(unsqueeze_970, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(mul_1114, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(unsqueeze_982, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    buf0 = empty((1, 19, 1, 1), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((1, 19, 1, 1), (19, 1, 19, 19), device='cpu', dtype=torch.float32)
    buf3 = empty((1, 25, 1, 1), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((1, 25, 1, 1), (25, 1, 25, 25), device='cpu', dtype=torch.float32)
    buf6 = empty((1, 30, 1, 1), device='cpu', dtype=torch.float32)
    buf7 = empty_strided((1, 30, 1, 1), (30, 1, 30, 30), device='cpu', dtype=torch.float32)
    buf9 = empty((1, 36, 1, 1), device='cpu', dtype=torch.float32)
    buf10 = empty_strided((1, 36, 1, 1), (36, 1, 36, 36), device='cpu', dtype=torch.float32)
    buf12 = empty((1, 42, 1, 1), device='cpu', dtype=torch.float32)
    buf13 = empty_strided((1, 42, 1, 1), (42, 1, 42, 42), device='cpu', dtype=torch.float32)
    buf15 = empty((1, 47, 1, 1), device='cpu', dtype=torch.float32)
    buf16 = empty_strided((1, 47, 1, 1), (47, 1, 47, 47), device='cpu', dtype=torch.float32)
    buf18 = empty((1, 53, 1, 1), device='cpu', dtype=torch.float32)
    buf19 = empty_strided((1, 53, 1, 1), (53, 1, 53, 53), device='cpu', dtype=torch.float32)
    buf21 = empty((1, 58, 1, 1), device='cpu', dtype=torch.float32)
    buf22 = empty_strided((1, 58, 1, 1), (58, 1, 58, 58), device='cpu', dtype=torch.float32)
    buf24 = empty((1, 64, 1, 1), device='cpu', dtype=torch.float32)
    buf25 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf27 = empty((1, 70, 1, 1), device='cpu', dtype=torch.float32)
    buf28 = empty_strided((1, 70, 1, 1), (70, 1, 70, 70), device='cpu', dtype=torch.float32)
    buf30 = empty((1, 75, 1, 1), device='cpu', dtype=torch.float32)
    buf31 = empty_strided((1, 75, 1, 1), (75, 1, 75, 75), device='cpu', dtype=torch.float32)
    buf33 = empty((1, 81, 1, 1), device='cpu', dtype=torch.float32)
    buf34 = empty_strided((1, 81, 1, 1), (81, 1, 81, 81), device='cpu', dtype=torch.float32)
    buf36 = empty((1, 87, 1, 1), device='cpu', dtype=torch.float32)
    buf37 = empty_strided((1, 87, 1, 1), (87, 1, 87, 87), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_0(c_void_p(convolution_11.data_ptr()), c_void_p(convolution_16.data_ptr()), c_void_p(convolution_21.data_ptr()), c_void_p(convolution_26.data_ptr()), c_void_p(convolution_31.data_ptr()), c_void_p(convolution_36.data_ptr()), c_void_p(convolution_41.data_ptr()), c_void_p(convolution_46.data_ptr()), c_void_p(convolution_51.data_ptr()), c_void_p(convolution_56.data_ptr()), c_void_p(convolution_61.data_ptr()), c_void_p(convolution_66.data_ptr()), c_void_p(convolution_71.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()))
    buf39 = empty((8, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_1, out=buf39)
    del permute_1
    buf40 = empty((1000, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_17, out=buf40)
    del clone_17
    buf41 = empty((1, 1000), device='cpu', dtype=torch.float32)
    buf42 = empty((1280, ), device='cpu', dtype=torch.float32)
    buf43 = empty((1280, ), device='cpu', dtype=torch.float32)
    buf44 = empty((1280, ), device='cpu', dtype=torch.float32)
    buf45 = empty((8, 1280, 7, 7), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_div_mul_native_batch_norm_backward_sum_1(c_void_p(tangents_1.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(mul_465.data_ptr()), c_void_p(convolution_74.data_ptr()), c_void_p(unsqueeze_250.data_ptr()), c_void_p(squeeze_184.data_ptr()), c_void_p(primals_97.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf45.data_ptr()))
    del buf39
    del buf43
    del convolution_74
    del mul_465
    del primals_97
    del squeeze_184
    del tangents_1
    del unsqueeze_250
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward]
    buf46 = aten.convolution_backward(buf45, cat_10, primals_225, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf45
    del cat_10
    del primals_225
    buf47 = buf46[0]
    buf48 = buf46[1]
    del buf46
    buf49 = empty((185, ), device='cpu', dtype=torch.float32)
    buf50 = empty((185, ), device='cpu', dtype=torch.float32)
    buf51 = empty((185, ), device='cpu', dtype=torch.float32)
    buf52 = empty((8, 185, 7, 7), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_2(c_void_p(buf47.data_ptr()), c_void_p(convolution_73.data_ptr()), c_void_p(unsqueeze_262.data_ptr()), c_void_p(squeeze_181.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf52.data_ptr()))
    del buf50
    del convolution_73
    del primals_95
    del squeeze_181
    del unsqueeze_262
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.slice_backward]
    buf53 = aten.convolution_backward(buf52, clamp_max_15, primals_224, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf52
    del clamp_max_15
    del primals_224
    buf54 = buf53[0]
    buf55 = buf53[1]
    del buf53
    buf56 = empty_strided((8, 1044, 1, 1), (1044, 1, 8352, 8352), device='cpu', dtype=torch.float32)
    buf57 = reinterpret_tensor(buf56, (8, 1044, 1, 1), (1044, 1, 1, 1), 0); del buf56  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_3(c_void_p(buf57.data_ptr()), c_void_p(add_304.data_ptr()), c_void_p(convolution_72.data_ptr()), c_void_p(buf54.data_ptr()))
    # Source Nodes: [sigmoid_12], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf58 = aten.convolution_backward(buf57, relu_12, primals_222, [1044], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf57
    del primals_222
    buf59 = buf58[0]
    buf60 = buf58[1]
    buf61 = buf58[2]
    del buf58
    buf62 = empty((87, ), device='cpu', dtype=torch.float32)
    buf63 = empty((87, ), device='cpu', dtype=torch.float32)
    buf64 = empty((87, ), device='cpu', dtype=torch.float32)
    buf65 = reinterpret_tensor(buf59, (8, 87, 1, 1), (87, 1, 1, 1), 0); del buf59  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_threshold_backward_4(c_void_p(buf65.data_ptr()), c_void_p(relu_12.data_ptr()), c_void_p(convolution_71.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(primals_220.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf64.data_ptr()))
    del buf36
    del buf37
    del buf63
    del convolution_71
    del primals_220
    del relu_12
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf66 = aten.convolution_backward(buf65, mean_12, primals_218, [87], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf65
    del mean_12
    del primals_218
    buf67 = buf66[0]
    buf68 = buf66[1]
    buf69 = buf66[2]
    del buf66
    buf70 = empty((1044, ), device='cpu', dtype=torch.float32)
    buf71 = empty((1044, ), device='cpu', dtype=torch.float32)
    buf72 = buf54; del buf54  # reuse
    buf73 = buf71; del buf71  # reuse
    buf74 = buf72; del buf72  # reuse
    cpp_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_5(c_void_p(buf74.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(add_304.data_ptr()), c_void_p(convolution_72.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(convolution_70.data_ptr()), c_void_p(unsqueeze_286.data_ptr()), c_void_p(squeeze_175.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(buf70.data_ptr()))
    del add_304
    del buf67
    del convolution_70
    del convolution_72
    del primals_93
    del squeeze_175
    del unsqueeze_286
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf75 = aten.convolution_backward(buf74, mul_433, primals_217, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1044, [True, True, False])
    del buf74
    del mul_433
    del primals_217
    buf76 = buf75[0]
    buf77 = buf75[1]
    del buf75
    buf78 = empty((1044, ), device='cpu', dtype=torch.float32)
    buf79 = empty((1044, ), device='cpu', dtype=torch.float32)
    buf80 = empty((1044, ), device='cpu', dtype=torch.float32)
    buf81 = buf76; del buf76  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_6(c_void_p(buf81.data_ptr()), c_void_p(mul_508.data_ptr()), c_void_p(convolution_69.data_ptr()), c_void_p(unsqueeze_298.data_ptr()), c_void_p(squeeze_172.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf80.data_ptr()))
    del buf79
    del convolution_69
    del mul_508
    del primals_91
    del squeeze_172
    del unsqueeze_298
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf82 = aten.convolution_backward(buf81, cat_9, primals_216, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf81
    del cat_9
    del primals_216
    buf83 = buf82[0]
    buf84 = buf82[1]
    del buf82
    buf85 = empty((174, ), device='cpu', dtype=torch.float32)
    buf86 = empty((174, ), device='cpu', dtype=torch.float32)
    buf87 = empty((174, ), device='cpu', dtype=torch.float32)
    buf88 = empty((8, 174, 7, 7), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_7(c_void_p(buf47.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(convolution_68.data_ptr()), c_void_p(unsqueeze_310.data_ptr()), c_void_p(squeeze_169.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf88.data_ptr()))
    del buf86
    del convolution_68
    del primals_89
    del squeeze_169
    del unsqueeze_310
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.slice_backward]
    buf89 = aten.convolution_backward(buf88, clamp_max_14, primals_215, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf88
    del clamp_max_14
    del primals_215
    buf90 = buf89[0]
    buf91 = buf89[1]
    del buf89
    buf92 = empty_strided((8, 972, 1, 1), (972, 1, 7776, 7776), device='cpu', dtype=torch.float32)
    buf93 = reinterpret_tensor(buf92, (8, 972, 1, 1), (972, 1, 1, 1), 0); del buf92  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_8(c_void_p(buf93.data_ptr()), c_void_p(add_283.data_ptr()), c_void_p(convolution_67.data_ptr()), c_void_p(buf90.data_ptr()))
    # Source Nodes: [sigmoid_11], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf94 = aten.convolution_backward(buf93, relu_11, primals_213, [972], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf93
    del primals_213
    buf95 = buf94[0]
    buf96 = buf94[1]
    buf97 = buf94[2]
    del buf94
    buf98 = empty((81, ), device='cpu', dtype=torch.float32)
    buf99 = empty((81, ), device='cpu', dtype=torch.float32)
    buf100 = empty((81, ), device='cpu', dtype=torch.float32)
    buf101 = reinterpret_tensor(buf95, (8, 81, 1, 1), (81, 1, 1, 1), 0); del buf95  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_threshold_backward_9(c_void_p(buf101.data_ptr()), c_void_p(relu_11.data_ptr()), c_void_p(convolution_66.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(primals_211.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf100.data_ptr()))
    del buf33
    del buf34
    del buf99
    del convolution_66
    del primals_211
    del relu_11
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf102 = aten.convolution_backward(buf101, mean_11, primals_209, [81], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf101
    del mean_11
    del primals_209
    buf103 = buf102[0]
    buf104 = buf102[1]
    buf105 = buf102[2]
    del buf102
    buf106 = empty((972, ), device='cpu', dtype=torch.float32)
    buf107 = empty((972, ), device='cpu', dtype=torch.float32)
    buf108 = buf90; del buf90  # reuse
    buf109 = buf107; del buf107  # reuse
    buf110 = buf108; del buf108  # reuse
    cpp_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_10(c_void_p(buf110.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(add_283.data_ptr()), c_void_p(convolution_67.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(convolution_65.data_ptr()), c_void_p(unsqueeze_334.data_ptr()), c_void_p(squeeze_163.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(buf106.data_ptr()))
    del add_283
    del buf103
    del convolution_65
    del convolution_67
    del primals_87
    del squeeze_163
    del unsqueeze_334
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf111 = aten.convolution_backward(buf110, mul_403, primals_208, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 972, [True, True, False])
    del buf110
    del mul_403
    del primals_208
    buf112 = buf111[0]
    buf113 = buf111[1]
    del buf111
    buf114 = empty((972, ), device='cpu', dtype=torch.float32)
    buf115 = empty((972, ), device='cpu', dtype=torch.float32)
    buf116 = empty((972, ), device='cpu', dtype=torch.float32)
    buf117 = buf112; del buf112  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_11(c_void_p(buf117.data_ptr()), c_void_p(mul_551.data_ptr()), c_void_p(convolution_64.data_ptr()), c_void_p(unsqueeze_346.data_ptr()), c_void_p(squeeze_160.data_ptr()), c_void_p(primals_85.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()))
    del buf115
    del convolution_64
    del mul_551
    del primals_85
    del squeeze_160
    del unsqueeze_346
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf118 = aten.convolution_backward(buf117, cat_8, primals_207, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf117
    del cat_8
    del primals_207
    buf119 = buf118[0]
    buf120 = buf118[1]
    del buf118
    buf121 = empty((162, ), device='cpu', dtype=torch.float32)
    buf122 = empty((162, ), device='cpu', dtype=torch.float32)
    buf123 = empty((8, 162, 7, 7), device='cpu', dtype=torch.float32)
    buf124 = buf122; del buf122  # reuse
    cpp_fused_add_native_batch_norm_backward_slice_backward_12(c_void_p(buf124.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(convolution_63.data_ptr()), c_void_p(unsqueeze_358.data_ptr()), c_void_p(squeeze_157.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf123.data_ptr()))
    del convolution_63
    del primals_83
    del squeeze_157
    del unsqueeze_358
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf125 = aten.convolution_backward(buf123, clamp_max_13, primals_206, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf123
    del clamp_max_13
    del primals_206
    buf126 = buf125[0]
    buf127 = buf125[1]
    del buf125
    buf128 = empty_strided((8, 906, 1, 1), (906, 1, 7248, 7248), device='cpu', dtype=torch.float32)
    buf129 = reinterpret_tensor(buf128, (8, 906, 1, 1), (906, 1, 1, 1), 0); del buf128  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_13(c_void_p(buf129.data_ptr()), c_void_p(add_262.data_ptr()), c_void_p(convolution_62.data_ptr()), c_void_p(buf126.data_ptr()))
    # Source Nodes: [sigmoid_10], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf130 = aten.convolution_backward(buf129, relu_10, primals_204, [906], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf129
    del primals_204
    buf131 = buf130[0]
    buf132 = buf130[1]
    buf133 = buf130[2]
    del buf130
    buf134 = empty((75, ), device='cpu', dtype=torch.float32)
    buf135 = empty((75, ), device='cpu', dtype=torch.float32)
    buf136 = empty((75, ), device='cpu', dtype=torch.float32)
    buf137 = reinterpret_tensor(buf131, (8, 75, 1, 1), (75, 1, 1, 1), 0); del buf131  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_threshold_backward_14(c_void_p(buf137.data_ptr()), c_void_p(relu_10.data_ptr()), c_void_p(convolution_61.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(primals_202.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf136.data_ptr()))
    del buf135
    del buf30
    del buf31
    del convolution_61
    del primals_202
    del relu_10
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf138 = aten.convolution_backward(buf137, mean_10, primals_200, [75], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf137
    del mean_10
    del primals_200
    buf139 = buf138[0]
    buf140 = buf138[1]
    buf141 = buf138[2]
    del buf138
    buf142 = empty((906, ), device='cpu', dtype=torch.float32)
    buf143 = empty((906, ), device='cpu', dtype=torch.float32)
    buf144 = buf126; del buf126  # reuse
    buf145 = buf143; del buf143  # reuse
    buf146 = buf144; del buf144  # reuse
    cpp_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_15(c_void_p(buf146.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(add_262.data_ptr()), c_void_p(convolution_62.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(convolution_60.data_ptr()), c_void_p(unsqueeze_382.data_ptr()), c_void_p(squeeze_151.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(buf142.data_ptr()))
    del add_262
    del buf139
    del convolution_60
    del convolution_62
    del primals_81
    del squeeze_151
    del unsqueeze_382
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf147 = aten.convolution_backward(buf146, mul_373, primals_199, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 906, [True, True, False])
    del buf146
    del mul_373
    del primals_199
    buf148 = buf147[0]
    buf149 = buf147[1]
    del buf147
    buf150 = empty((906, ), device='cpu', dtype=torch.float32)
    buf151 = empty((906, ), device='cpu', dtype=torch.float32)
    buf152 = empty((906, ), device='cpu', dtype=torch.float32)
    buf153 = buf148; del buf148  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_16(c_void_p(buf153.data_ptr()), c_void_p(mul_594.data_ptr()), c_void_p(convolution_59.data_ptr()), c_void_p(unsqueeze_394.data_ptr()), c_void_p(squeeze_148.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf152.data_ptr()))
    del buf151
    del convolution_59
    del mul_594
    del primals_79
    del squeeze_148
    del unsqueeze_394
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf154 = aten.convolution_backward(buf153, cat_7, primals_198, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf153
    del cat_7
    del primals_198
    buf155 = buf154[0]
    buf156 = buf154[1]
    del buf154
    buf157 = empty((151, ), device='cpu', dtype=torch.float32)
    buf158 = empty((151, ), device='cpu', dtype=torch.float32)
    buf159 = empty_strided((8, 151, 7, 7), (7399, 1, 1057, 151), device='cpu', dtype=torch.float32)
    buf160 = buf158; del buf158  # reuse
    buf161 = empty((8, 151, 7, 7), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_17(c_void_p(buf160.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(convolution_58.data_ptr()), c_void_p(unsqueeze_406.data_ptr()), c_void_p(squeeze_145.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf161.data_ptr()))
    del buf159
    del convolution_58
    del primals_77
    del squeeze_145
    del unsqueeze_406
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf162 = aten.convolution_backward(buf161, clamp_max_12, primals_197, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf161
    del clamp_max_12
    del primals_197
    buf163 = buf162[0]
    buf164 = buf162[1]
    del buf162
    buf165 = empty_strided((8, 840, 1, 1), (840, 1, 6720, 6720), device='cpu', dtype=torch.float32)
    buf166 = reinterpret_tensor(buf165, (8, 840, 1, 1), (840, 1, 1, 1), 0); del buf165  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_18(c_void_p(buf166.data_ptr()), c_void_p(add_241.data_ptr()), c_void_p(convolution_57.data_ptr()), c_void_p(buf163.data_ptr()))
    # Source Nodes: [sigmoid_9], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf167 = aten.convolution_backward(buf166, relu_9, primals_195, [840], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf166
    del primals_195
    buf168 = buf167[0]
    buf169 = buf167[1]
    buf170 = buf167[2]
    del buf167
    buf171 = empty((70, ), device='cpu', dtype=torch.float32)
    buf172 = empty((70, ), device='cpu', dtype=torch.float32)
    buf173 = empty((70, ), device='cpu', dtype=torch.float32)
    buf174 = reinterpret_tensor(buf168, (8, 70, 1, 1), (70, 1, 1, 1), 0); del buf168  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_threshold_backward_19(c_void_p(buf174.data_ptr()), c_void_p(relu_9.data_ptr()), c_void_p(convolution_56.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(primals_193.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf173.data_ptr()))
    del buf172
    del buf27
    del buf28
    del convolution_56
    del primals_193
    del relu_9
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf175 = aten.convolution_backward(buf174, mean_9, primals_191, [70], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf174
    del mean_9
    del primals_191
    buf176 = buf175[0]
    buf177 = buf175[1]
    buf178 = buf175[2]
    del buf175
    buf179 = empty((840, ), device='cpu', dtype=torch.float32)
    buf180 = empty((840, ), device='cpu', dtype=torch.float32)
    buf181 = buf163; del buf163  # reuse
    buf182 = buf180; del buf180  # reuse
    buf183 = buf181; del buf181  # reuse
    cpp_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_20(c_void_p(buf183.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(add_241.data_ptr()), c_void_p(convolution_57.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(convolution_55.data_ptr()), c_void_p(unsqueeze_430.data_ptr()), c_void_p(squeeze_139.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(buf179.data_ptr()))
    del add_241
    del buf176
    del convolution_55
    del convolution_57
    del primals_75
    del squeeze_139
    del unsqueeze_430
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf184 = aten.convolution_backward(buf183, mul_343, primals_190, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 840, [True, True, False])
    del buf183
    del mul_343
    del primals_190
    buf185 = buf184[0]
    buf186 = buf184[1]
    del buf184
    buf187 = empty((840, ), device='cpu', dtype=torch.float32)
    buf188 = empty((840, ), device='cpu', dtype=torch.float32)
    buf189 = empty((840, ), device='cpu', dtype=torch.float32)
    buf190 = buf185; del buf185  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_21(c_void_p(buf190.data_ptr()), c_void_p(mul_637.data_ptr()), c_void_p(convolution_54.data_ptr()), c_void_p(unsqueeze_442.data_ptr()), c_void_p(squeeze_136.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf189.data_ptr()))
    del buf188
    del convolution_54
    del mul_637
    del primals_73
    del squeeze_136
    del unsqueeze_442
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf191 = aten.convolution_backward(buf190, add_231, primals_189, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_231
    del buf190
    del primals_189
    buf192 = buf191[0]
    buf193 = buf191[1]
    del buf191
    buf194 = buf192; del buf192  # reuse
    buf195 = empty((140, ), device='cpu', dtype=torch.float32)
    buf196 = empty((140, ), device='cpu', dtype=torch.float32)
    buf197 = empty((140, ), device='cpu', dtype=torch.float32)
    buf198 = buf194; del buf194  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_22(c_void_p(buf198.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(convolution_53.data_ptr()), c_void_p(unsqueeze_454.data_ptr()), c_void_p(squeeze_133.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf197.data_ptr()))
    del buf119
    del buf155
    del buf196
    del buf47
    del buf83
    del convolution_53
    del primals_71
    del squeeze_133
    del unsqueeze_454
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf199 = aten.convolution_backward(buf198, clamp_max_11, primals_188, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf198
    del clamp_max_11
    del primals_188
    buf200 = buf199[0]
    buf201 = buf199[1]
    del buf199
    buf202 = empty_strided((8, 768, 1, 1), (768, 1, 6144, 6144), device='cpu', dtype=torch.float32)
    buf203 = reinterpret_tensor(buf202, (8, 768, 1, 1), (768, 1, 1, 1), 0); del buf202  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_23(c_void_p(buf203.data_ptr()), c_void_p(add_221.data_ptr()), c_void_p(convolution_52.data_ptr()), c_void_p(buf200.data_ptr()))
    # Source Nodes: [sigmoid_8], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf204 = aten.convolution_backward(buf203, relu_8, primals_186, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf203
    del primals_186
    buf205 = buf204[0]
    buf206 = buf204[1]
    buf207 = buf204[2]
    del buf204
    buf208 = empty((64, ), device='cpu', dtype=torch.float32)
    buf209 = empty((64, ), device='cpu', dtype=torch.float32)
    buf210 = empty((64, ), device='cpu', dtype=torch.float32)
    buf211 = reinterpret_tensor(buf205, (8, 64, 1, 1), (64, 1, 1, 1), 0); del buf205  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_threshold_backward_24(c_void_p(buf211.data_ptr()), c_void_p(relu_8.data_ptr()), c_void_p(convolution_51.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(primals_184.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf210.data_ptr()))
    del buf209
    del buf24
    del buf25
    del convolution_51
    del primals_184
    del relu_8
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf212 = aten.convolution_backward(buf211, mean_8, primals_182, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf211
    del mean_8
    del primals_182
    buf213 = buf212[0]
    buf214 = buf212[1]
    buf215 = buf212[2]
    del buf212
    buf216 = empty((768, ), device='cpu', dtype=torch.float32)
    buf217 = empty((768, ), device='cpu', dtype=torch.float32)
    buf218 = buf200; del buf200  # reuse
    buf219 = buf217; del buf217  # reuse
    buf220 = buf218; del buf218  # reuse
    cpp_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_25(c_void_p(buf220.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(add_221.data_ptr()), c_void_p(convolution_52.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(convolution_50.data_ptr()), c_void_p(unsqueeze_478.data_ptr()), c_void_p(squeeze_127.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(buf216.data_ptr()))
    del add_221
    del buf213
    del convolution_50
    del convolution_52
    del primals_69
    del squeeze_127
    del unsqueeze_478
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf221 = aten.convolution_backward(buf220, mul_313, primals_181, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
    del buf220
    del mul_313
    del primals_181
    buf222 = buf221[0]
    buf223 = buf221[1]
    del buf221
    buf224 = empty((768, ), device='cpu', dtype=torch.float32)
    buf225 = empty((768, ), device='cpu', dtype=torch.float32)
    buf226 = empty((768, ), device='cpu', dtype=torch.float32)
    buf227 = buf222; del buf222  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_26(c_void_p(buf227.data_ptr()), c_void_p(mul_680.data_ptr()), c_void_p(convolution_49.data_ptr()), c_void_p(unsqueeze_490.data_ptr()), c_void_p(squeeze_124.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf226.data_ptr()))
    del buf225
    del convolution_49
    del mul_680
    del primals_67
    del squeeze_124
    del unsqueeze_490
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf228 = aten.convolution_backward(buf227, cat_6, primals_180, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf227
    del cat_6
    del primals_180
    buf229 = buf228[0]
    buf230 = buf228[1]
    del buf228
    buf231 = empty((128, ), device='cpu', dtype=torch.float32)
    buf232 = empty((128, ), device='cpu', dtype=torch.float32)
    buf233 = empty((128, ), device='cpu', dtype=torch.float32)
    buf234 = empty((8, 128, 14, 14), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_27(c_void_p(buf229.data_ptr()), c_void_p(convolution_48.data_ptr()), c_void_p(unsqueeze_502.data_ptr()), c_void_p(squeeze_121.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(buf234.data_ptr()))
    del buf232
    del convolution_48
    del primals_65
    del squeeze_121
    del unsqueeze_502
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.slice_backward]
    buf235 = aten.convolution_backward(buf234, clamp_max_10, primals_179, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf234
    del clamp_max_10
    del primals_179
    buf236 = buf235[0]
    buf237 = buf235[1]
    del buf235
    buf238 = empty_strided((8, 702, 1, 1), (702, 1, 5616, 5616), device='cpu', dtype=torch.float32)
    buf239 = reinterpret_tensor(buf238, (8, 702, 1, 1), (702, 1, 1, 1), 0); del buf238  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_28(c_void_p(buf239.data_ptr()), c_void_p(add_200.data_ptr()), c_void_p(convolution_47.data_ptr()), c_void_p(buf236.data_ptr()))
    # Source Nodes: [sigmoid_7], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf240 = aten.convolution_backward(buf239, relu_7, primals_177, [702], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf239
    del primals_177
    buf241 = buf240[0]
    buf242 = buf240[1]
    buf243 = buf240[2]
    del buf240
    buf244 = empty((58, ), device='cpu', dtype=torch.float32)
    buf245 = empty((58, ), device='cpu', dtype=torch.float32)
    buf246 = empty((58, ), device='cpu', dtype=torch.float32)
    buf247 = reinterpret_tensor(buf241, (8, 58, 1, 1), (58, 1, 1, 1), 0); del buf241  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_threshold_backward_29(c_void_p(buf247.data_ptr()), c_void_p(relu_7.data_ptr()), c_void_p(convolution_46.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(primals_175.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf246.data_ptr()))
    del buf21
    del buf22
    del buf245
    del convolution_46
    del primals_175
    del relu_7
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf248 = aten.convolution_backward(buf247, mean_7, primals_173, [58], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf247
    del mean_7
    del primals_173
    buf249 = buf248[0]
    buf250 = buf248[1]
    buf251 = buf248[2]
    del buf248
    buf252 = empty((702, ), device='cpu', dtype=torch.float32)
    buf253 = empty((702, ), device='cpu', dtype=torch.float32)
    buf254 = buf236; del buf236  # reuse
    buf255 = buf253; del buf253  # reuse
    buf256 = buf254; del buf254  # reuse
    cpp_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_30(c_void_p(buf256.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(add_200.data_ptr()), c_void_p(convolution_47.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(convolution_45.data_ptr()), c_void_p(unsqueeze_526.data_ptr()), c_void_p(squeeze_115.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(buf252.data_ptr()))
    del add_200
    del buf249
    del convolution_45
    del convolution_47
    del primals_63
    del squeeze_115
    del unsqueeze_526
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf257 = aten.convolution_backward(buf256, mul_283, primals_172, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 702, [True, True, False])
    del buf256
    del mul_283
    del primals_172
    buf258 = buf257[0]
    buf259 = buf257[1]
    del buf257
    buf260 = empty((702, ), device='cpu', dtype=torch.float32)
    buf261 = empty((702, ), device='cpu', dtype=torch.float32)
    buf262 = empty((702, ), device='cpu', dtype=torch.float32)
    buf263 = buf258; del buf258  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_31(c_void_p(buf263.data_ptr()), c_void_p(mul_723.data_ptr()), c_void_p(convolution_44.data_ptr()), c_void_p(unsqueeze_538.data_ptr()), c_void_p(squeeze_112.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf262.data_ptr()))
    del buf261
    del convolution_44
    del mul_723
    del primals_61
    del squeeze_112
    del unsqueeze_538
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf264 = aten.convolution_backward(buf263, cat_5, primals_171, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf263
    del cat_5
    del primals_171
    buf265 = buf264[0]
    buf266 = buf264[1]
    del buf264
    buf267 = empty((117, ), device='cpu', dtype=torch.float32)
    buf268 = empty((117, ), device='cpu', dtype=torch.float32)
    buf269 = empty((117, ), device='cpu', dtype=torch.float32)
    buf270 = empty((8, 117, 14, 14), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_32(c_void_p(buf229.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(convolution_43.data_ptr()), c_void_p(unsqueeze_550.data_ptr()), c_void_p(squeeze_109.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf270.data_ptr()))
    del buf268
    del convolution_43
    del primals_59
    del squeeze_109
    del unsqueeze_550
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.slice_backward]
    buf271 = aten.convolution_backward(buf270, clamp_max_9, primals_170, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf270
    del clamp_max_9
    del primals_170
    buf272 = buf271[0]
    buf273 = buf271[1]
    del buf271
    buf274 = empty_strided((8, 636, 1, 1), (636, 1, 5088, 5088), device='cpu', dtype=torch.float32)
    buf275 = reinterpret_tensor(buf274, (8, 636, 1, 1), (636, 1, 1, 1), 0); del buf274  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_33(c_void_p(buf275.data_ptr()), c_void_p(add_179.data_ptr()), c_void_p(convolution_42.data_ptr()), c_void_p(buf272.data_ptr()))
    # Source Nodes: [sigmoid_6], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf276 = aten.convolution_backward(buf275, relu_6, primals_168, [636], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf275
    del primals_168
    buf277 = buf276[0]
    buf278 = buf276[1]
    buf279 = buf276[2]
    del buf276
    buf280 = empty((53, ), device='cpu', dtype=torch.float32)
    buf281 = empty((53, ), device='cpu', dtype=torch.float32)
    buf282 = empty((53, ), device='cpu', dtype=torch.float32)
    buf283 = reinterpret_tensor(buf277, (8, 53, 1, 1), (53, 1, 1, 1), 0); del buf277  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_threshold_backward_34(c_void_p(buf283.data_ptr()), c_void_p(relu_6.data_ptr()), c_void_p(convolution_41.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(primals_166.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf282.data_ptr()))
    del buf18
    del buf19
    del buf281
    del convolution_41
    del primals_166
    del relu_6
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf284 = aten.convolution_backward(buf283, mean_6, primals_164, [53], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf283
    del mean_6
    del primals_164
    buf285 = buf284[0]
    buf286 = buf284[1]
    buf287 = buf284[2]
    del buf284
    buf288 = empty((636, ), device='cpu', dtype=torch.float32)
    buf289 = empty((636, ), device='cpu', dtype=torch.float32)
    buf290 = buf272; del buf272  # reuse
    buf291 = buf289; del buf289  # reuse
    buf292 = buf290; del buf290  # reuse
    cpp_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_35(c_void_p(buf292.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(add_179.data_ptr()), c_void_p(convolution_42.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(convolution_40.data_ptr()), c_void_p(unsqueeze_574.data_ptr()), c_void_p(squeeze_103.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(buf288.data_ptr()))
    del add_179
    del buf285
    del convolution_40
    del convolution_42
    del primals_57
    del squeeze_103
    del unsqueeze_574
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf293 = aten.convolution_backward(buf292, mul_253, primals_163, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 636, [True, True, False])
    del buf292
    del mul_253
    del primals_163
    buf294 = buf293[0]
    buf295 = buf293[1]
    del buf293
    buf296 = empty((636, ), device='cpu', dtype=torch.float32)
    buf297 = empty((636, ), device='cpu', dtype=torch.float32)
    buf298 = empty((636, ), device='cpu', dtype=torch.float32)
    buf299 = buf294; del buf294  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_36(c_void_p(buf299.data_ptr()), c_void_p(mul_766.data_ptr()), c_void_p(convolution_39.data_ptr()), c_void_p(unsqueeze_586.data_ptr()), c_void_p(squeeze_100.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf298.data_ptr()))
    del buf297
    del convolution_39
    del mul_766
    del primals_55
    del squeeze_100
    del unsqueeze_586
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf300 = aten.convolution_backward(buf299, cat_4, primals_162, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf299
    del cat_4
    del primals_162
    buf301 = buf300[0]
    buf302 = buf300[1]
    del buf300
    buf303 = empty((106, ), device='cpu', dtype=torch.float32)
    buf304 = empty((106, ), device='cpu', dtype=torch.float32)
    buf305 = empty((8, 106, 14, 14), device='cpu', dtype=torch.float32)
    buf306 = buf304; del buf304  # reuse
    cpp_fused_add_native_batch_norm_backward_slice_backward_37(c_void_p(buf306.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(convolution_38.data_ptr()), c_void_p(unsqueeze_598.data_ptr()), c_void_p(squeeze_97.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf305.data_ptr()))
    del convolution_38
    del primals_53
    del squeeze_97
    del unsqueeze_598
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf307 = aten.convolution_backward(buf305, clamp_max_8, primals_161, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf305
    del clamp_max_8
    del primals_161
    buf308 = buf307[0]
    buf309 = buf307[1]
    del buf307
    buf310 = empty_strided((8, 570, 1, 1), (570, 1, 4560, 4560), device='cpu', dtype=torch.float32)
    buf311 = reinterpret_tensor(buf310, (8, 570, 1, 1), (570, 1, 1, 1), 0); del buf310  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_38(c_void_p(buf311.data_ptr()), c_void_p(add_158.data_ptr()), c_void_p(convolution_37.data_ptr()), c_void_p(buf308.data_ptr()))
    # Source Nodes: [sigmoid_5], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf312 = aten.convolution_backward(buf311, relu_5, primals_159, [570], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf311
    del primals_159
    buf313 = buf312[0]
    buf314 = buf312[1]
    buf315 = buf312[2]
    del buf312
    buf316 = empty((47, ), device='cpu', dtype=torch.float32)
    buf317 = empty((47, ), device='cpu', dtype=torch.float32)
    buf318 = empty((47, ), device='cpu', dtype=torch.float32)
    buf319 = reinterpret_tensor(buf313, (8, 47, 1, 1), (47, 1, 1, 1), 0); del buf313  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_threshold_backward_39(c_void_p(buf319.data_ptr()), c_void_p(relu_5.data_ptr()), c_void_p(convolution_36.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(primals_157.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf318.data_ptr()))
    del buf15
    del buf16
    del buf317
    del convolution_36
    del primals_157
    del relu_5
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf320 = aten.convolution_backward(buf319, mean_5, primals_155, [47], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf319
    del mean_5
    del primals_155
    buf321 = buf320[0]
    buf322 = buf320[1]
    buf323 = buf320[2]
    del buf320
    buf324 = empty((570, ), device='cpu', dtype=torch.float32)
    buf325 = empty((570, ), device='cpu', dtype=torch.float32)
    buf326 = buf308; del buf308  # reuse
    buf327 = buf325; del buf325  # reuse
    buf328 = buf326; del buf326  # reuse
    cpp_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_40(c_void_p(buf328.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(add_158.data_ptr()), c_void_p(convolution_37.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(convolution_35.data_ptr()), c_void_p(unsqueeze_622.data_ptr()), c_void_p(squeeze_91.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(buf324.data_ptr()))
    del add_158
    del buf321
    del convolution_35
    del convolution_37
    del primals_51
    del squeeze_91
    del unsqueeze_622
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf329 = aten.convolution_backward(buf328, mul_223, primals_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 570, [True, True, False])
    del buf328
    del mul_223
    del primals_154
    buf330 = buf329[0]
    buf331 = buf329[1]
    del buf329
    buf332 = empty((570, ), device='cpu', dtype=torch.float32)
    buf333 = empty((570, ), device='cpu', dtype=torch.float32)
    buf334 = empty((570, ), device='cpu', dtype=torch.float32)
    buf335 = buf330; del buf330  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_41(c_void_p(buf335.data_ptr()), c_void_p(mul_809.data_ptr()), c_void_p(convolution_34.data_ptr()), c_void_p(unsqueeze_634.data_ptr()), c_void_p(squeeze_88.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf334.data_ptr()))
    del buf333
    del convolution_34
    del mul_809
    del primals_49
    del squeeze_88
    del unsqueeze_634
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf336 = aten.convolution_backward(buf335, cat_3, primals_153, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf335
    del cat_3
    del primals_153
    buf337 = buf336[0]
    buf338 = buf336[1]
    del buf336
    buf339 = empty((95, ), device='cpu', dtype=torch.float32)
    buf340 = empty((95, ), device='cpu', dtype=torch.float32)
    buf341 = empty_strided((8, 95, 14, 14), (18620, 1, 1330, 95), device='cpu', dtype=torch.float32)
    buf342 = buf340; del buf340  # reuse
    buf343 = empty((8, 95, 14, 14), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_42(c_void_p(buf342.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(convolution_33.data_ptr()), c_void_p(unsqueeze_646.data_ptr()), c_void_p(squeeze_85.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf343.data_ptr()))
    del buf341
    del convolution_33
    del primals_47
    del squeeze_85
    del unsqueeze_646
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf344 = aten.convolution_backward(buf343, clamp_max_7, primals_152, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf343
    del clamp_max_7
    del primals_152
    buf345 = buf344[0]
    buf346 = buf344[1]
    del buf344
    buf347 = empty_strided((8, 504, 1, 1), (504, 1, 4032, 4032), device='cpu', dtype=torch.float32)
    buf348 = reinterpret_tensor(buf347, (8, 504, 1, 1), (504, 1, 1, 1), 0); del buf347  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_43(c_void_p(buf348.data_ptr()), c_void_p(add_137.data_ptr()), c_void_p(convolution_32.data_ptr()), c_void_p(buf345.data_ptr()))
    # Source Nodes: [sigmoid_4], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf349 = aten.convolution_backward(buf348, relu_4, primals_150, [504], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf348
    del primals_150
    buf350 = buf349[0]
    buf351 = buf349[1]
    buf352 = buf349[2]
    del buf349
    buf353 = empty((42, ), device='cpu', dtype=torch.float32)
    buf354 = empty((42, ), device='cpu', dtype=torch.float32)
    buf355 = empty((42, ), device='cpu', dtype=torch.float32)
    buf356 = reinterpret_tensor(buf350, (8, 42, 1, 1), (42, 1, 1, 1), 0); del buf350  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_threshold_backward_44(c_void_p(buf356.data_ptr()), c_void_p(relu_4.data_ptr()), c_void_p(convolution_31.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(primals_148.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf355.data_ptr()))
    del buf12
    del buf13
    del buf354
    del convolution_31
    del primals_148
    del relu_4
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf357 = aten.convolution_backward(buf356, mean_4, primals_146, [42], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf356
    del mean_4
    del primals_146
    buf358 = buf357[0]
    buf359 = buf357[1]
    buf360 = buf357[2]
    del buf357
    buf361 = empty((504, ), device='cpu', dtype=torch.float32)
    buf362 = empty((504, ), device='cpu', dtype=torch.float32)
    buf363 = buf345; del buf345  # reuse
    buf364 = buf362; del buf362  # reuse
    buf365 = buf363; del buf363  # reuse
    cpp_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_45(c_void_p(buf365.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(add_137.data_ptr()), c_void_p(convolution_32.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(convolution_30.data_ptr()), c_void_p(unsqueeze_670.data_ptr()), c_void_p(squeeze_79.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(buf361.data_ptr()))
    del add_137
    del buf358
    del convolution_30
    del convolution_32
    del primals_45
    del squeeze_79
    del unsqueeze_670
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf366 = aten.convolution_backward(buf365, mul_193, primals_145, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 504, [True, True, False])
    del buf365
    del mul_193
    del primals_145
    buf367 = buf366[0]
    buf368 = buf366[1]
    del buf366
    buf369 = empty((504, ), device='cpu', dtype=torch.float32)
    buf370 = empty((504, ), device='cpu', dtype=torch.float32)
    buf371 = empty((504, ), device='cpu', dtype=torch.float32)
    buf372 = buf367; del buf367  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_46(c_void_p(buf372.data_ptr()), c_void_p(mul_852.data_ptr()), c_void_p(convolution_29.data_ptr()), c_void_p(unsqueeze_682.data_ptr()), c_void_p(squeeze_76.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf371.data_ptr()))
    del buf370
    del convolution_29
    del mul_852
    del primals_43
    del squeeze_76
    del unsqueeze_682
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf373 = aten.convolution_backward(buf372, cat_2, primals_144, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf372
    del cat_2
    del primals_144
    buf374 = buf373[0]
    buf375 = buf373[1]
    del buf373
    buf376 = buf374; del buf374  # reuse
    buf377 = empty((84, ), device='cpu', dtype=torch.float32)
    buf378 = empty((84, ), device='cpu', dtype=torch.float32)
    buf379 = empty((84, ), device='cpu', dtype=torch.float32)
    buf380 = empty((8, 84, 14, 14), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_47(c_void_p(buf376.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(convolution_28.data_ptr()), c_void_p(unsqueeze_694.data_ptr()), c_void_p(squeeze_73.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf380.data_ptr()))
    del buf229
    del buf265
    del buf301
    del buf337
    del buf378
    del convolution_28
    del primals_41
    del squeeze_73
    del unsqueeze_694
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.slice_backward]
    buf381 = aten.convolution_backward(buf380, clamp_max_6, primals_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf380
    del clamp_max_6
    del primals_143
    buf382 = buf381[0]
    buf383 = buf381[1]
    del buf381
    buf384 = empty_strided((8, 432, 1, 1), (432, 1, 3456, 3456), device='cpu', dtype=torch.float32)
    buf385 = reinterpret_tensor(buf384, (8, 432, 1, 1), (432, 1, 1, 1), 0); del buf384  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_48(c_void_p(buf385.data_ptr()), c_void_p(add_116.data_ptr()), c_void_p(convolution_27.data_ptr()), c_void_p(buf382.data_ptr()))
    # Source Nodes: [sigmoid_3], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf386 = aten.convolution_backward(buf385, relu_3, primals_141, [432], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf385
    del primals_141
    buf387 = buf386[0]
    buf388 = buf386[1]
    buf389 = buf386[2]
    del buf386
    buf390 = empty((36, ), device='cpu', dtype=torch.float32)
    buf391 = empty((36, ), device='cpu', dtype=torch.float32)
    buf392 = empty((36, ), device='cpu', dtype=torch.float32)
    buf393 = reinterpret_tensor(buf387, (8, 36, 1, 1), (36, 1, 1, 1), 0); del buf387  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_threshold_backward_49(c_void_p(buf393.data_ptr()), c_void_p(relu_3.data_ptr()), c_void_p(convolution_26.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(primals_139.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf392.data_ptr()))
    del buf10
    del buf391
    del buf9
    del convolution_26
    del primals_139
    del relu_3
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf394 = aten.convolution_backward(buf393, mean_3, primals_137, [36], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf393
    del mean_3
    del primals_137
    buf395 = buf394[0]
    buf396 = buf394[1]
    buf397 = buf394[2]
    del buf394
    buf398 = empty((432, ), device='cpu', dtype=torch.float32)
    buf399 = empty((432, ), device='cpu', dtype=torch.float32)
    buf400 = buf382; del buf382  # reuse
    buf401 = buf399; del buf399  # reuse
    buf402 = buf400; del buf400  # reuse
    cpp_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_50(c_void_p(buf402.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(add_116.data_ptr()), c_void_p(convolution_27.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(convolution_25.data_ptr()), c_void_p(unsqueeze_718.data_ptr()), c_void_p(squeeze_67.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(buf398.data_ptr()))
    del add_116
    del buf395
    del convolution_25
    del convolution_27
    del primals_39
    del squeeze_67
    del unsqueeze_718
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf403 = aten.convolution_backward(buf402, mul_163, primals_136, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 432, [True, True, False])
    del buf402
    del mul_163
    del primals_136
    buf404 = buf403[0]
    buf405 = buf403[1]
    del buf403
    buf406 = empty((432, ), device='cpu', dtype=torch.float32)
    buf407 = empty((432, ), device='cpu', dtype=torch.float32)
    buf408 = empty((432, ), device='cpu', dtype=torch.float32)
    buf409 = buf404; del buf404  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_51(c_void_p(buf409.data_ptr()), c_void_p(mul_895.data_ptr()), c_void_p(convolution_24.data_ptr()), c_void_p(unsqueeze_730.data_ptr()), c_void_p(squeeze_64.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(buf408.data_ptr()))
    del buf407
    del convolution_24
    del mul_895
    del primals_37
    del squeeze_64
    del unsqueeze_730
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf410 = aten.convolution_backward(buf409, add_106, primals_135, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_106
    del buf409
    del primals_135
    buf411 = buf410[0]
    buf412 = buf410[1]
    del buf410
    buf413 = empty((72, ), device='cpu', dtype=torch.float32)
    buf414 = empty((72, ), device='cpu', dtype=torch.float32)
    buf415 = empty((72, ), device='cpu', dtype=torch.float32)
    buf416 = buf411; del buf411  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_52(c_void_p(buf416.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(unsqueeze_742.data_ptr()), c_void_p(squeeze_61.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf415.data_ptr()))
    del buf376
    del buf414
    del convolution_23
    del primals_35
    del squeeze_61
    del unsqueeze_742
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf417 = aten.convolution_backward(buf416, clamp_max_5, primals_134, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf416
    del clamp_max_5
    del primals_134
    buf418 = buf417[0]
    buf419 = buf417[1]
    del buf417
    buf420 = empty_strided((8, 366, 1, 1), (366, 1, 2928, 2928), device='cpu', dtype=torch.float32)
    buf421 = reinterpret_tensor(buf420, (8, 366, 1, 1), (366, 1, 1, 1), 0); del buf420  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_53(c_void_p(buf421.data_ptr()), c_void_p(add_96.data_ptr()), c_void_p(convolution_22.data_ptr()), c_void_p(buf418.data_ptr()))
    # Source Nodes: [sigmoid_2], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf422 = aten.convolution_backward(buf421, relu_2, primals_132, [366], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf421
    del primals_132
    buf423 = buf422[0]
    buf424 = buf422[1]
    buf425 = buf422[2]
    del buf422
    buf426 = empty((30, ), device='cpu', dtype=torch.float32)
    buf427 = empty((30, ), device='cpu', dtype=torch.float32)
    buf428 = empty((30, ), device='cpu', dtype=torch.float32)
    buf429 = reinterpret_tensor(buf423, (8, 30, 1, 1), (30, 1, 1, 1), 0); del buf423  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_threshold_backward_54(c_void_p(buf429.data_ptr()), c_void_p(relu_2.data_ptr()), c_void_p(convolution_21.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(primals_130.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf428.data_ptr()))
    del buf427
    del buf6
    del buf7
    del convolution_21
    del primals_130
    del relu_2
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf430 = aten.convolution_backward(buf429, mean_2, primals_128, [30], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf429
    del mean_2
    del primals_128
    buf431 = buf430[0]
    buf432 = buf430[1]
    buf433 = buf430[2]
    del buf430
    buf434 = empty((366, ), device='cpu', dtype=torch.float32)
    buf435 = empty((366, ), device='cpu', dtype=torch.float32)
    buf436 = buf418; del buf418  # reuse
    buf437 = buf435; del buf435  # reuse
    buf438 = buf436; del buf436  # reuse
    cpp_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_55(c_void_p(buf438.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(add_96.data_ptr()), c_void_p(convolution_22.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(convolution_20.data_ptr()), c_void_p(unsqueeze_766.data_ptr()), c_void_p(squeeze_55.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf434.data_ptr()))
    del add_96
    del buf431
    del convolution_20
    del convolution_22
    del primals_33
    del squeeze_55
    del unsqueeze_766
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf439 = aten.convolution_backward(buf438, mul_133, primals_127, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 366, [True, True, False])
    del buf438
    del mul_133
    del primals_127
    buf440 = buf439[0]
    buf441 = buf439[1]
    del buf439
    buf442 = empty((366, ), device='cpu', dtype=torch.float32)
    buf443 = empty((366, ), device='cpu', dtype=torch.float32)
    buf444 = empty((366, ), device='cpu', dtype=torch.float32)
    buf445 = buf440; del buf440  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_56(c_void_p(buf445.data_ptr()), c_void_p(mul_938.data_ptr()), c_void_p(convolution_19.data_ptr()), c_void_p(unsqueeze_778.data_ptr()), c_void_p(squeeze_52.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf444.data_ptr()))
    del buf443
    del convolution_19
    del mul_938
    del primals_31
    del squeeze_52
    del unsqueeze_778
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf446 = aten.convolution_backward(buf445, cat_1, primals_126, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf445
    del cat_1
    del primals_126
    buf447 = buf446[0]
    buf448 = buf446[1]
    del buf446
    buf449 = empty((61, ), device='cpu', dtype=torch.float32)
    buf450 = empty((61, ), device='cpu', dtype=torch.float32)
    buf451 = empty((61, ), device='cpu', dtype=torch.float32)
    buf452 = empty((8, 61, 28, 28), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_57(c_void_p(buf447.data_ptr()), c_void_p(convolution_18.data_ptr()), c_void_p(unsqueeze_790.data_ptr()), c_void_p(squeeze_49.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(buf452.data_ptr()))
    del buf450
    del convolution_18
    del primals_29
    del squeeze_49
    del unsqueeze_790
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.slice_backward]
    buf453 = aten.convolution_backward(buf452, clamp_max_4, primals_125, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf452
    del clamp_max_4
    del primals_125
    buf454 = buf453[0]
    buf455 = buf453[1]
    del buf453
    buf456 = empty_strided((8, 300, 1, 1), (300, 1, 2400, 2400), device='cpu', dtype=torch.float32)
    buf457 = reinterpret_tensor(buf456, (8, 300, 1, 1), (300, 1, 1, 1), 0); del buf456  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_58(c_void_p(buf457.data_ptr()), c_void_p(add_75.data_ptr()), c_void_p(convolution_17.data_ptr()), c_void_p(buf454.data_ptr()))
    # Source Nodes: [sigmoid_1], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf458 = aten.convolution_backward(buf457, relu_1, primals_123, [300], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf457
    del primals_123
    buf459 = buf458[0]
    buf460 = buf458[1]
    buf461 = buf458[2]
    del buf458
    buf462 = empty((25, ), device='cpu', dtype=torch.float32)
    buf463 = empty((25, ), device='cpu', dtype=torch.float32)
    buf464 = empty((25, ), device='cpu', dtype=torch.float32)
    buf465 = reinterpret_tensor(buf459, (8, 25, 1, 1), (25, 1, 1, 1), 0); del buf459  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_threshold_backward_59(c_void_p(buf465.data_ptr()), c_void_p(relu_1.data_ptr()), c_void_p(convolution_16.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(primals_121.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(buf463.data_ptr()), c_void_p(buf464.data_ptr()))
    del buf3
    del buf4
    del buf463
    del convolution_16
    del primals_121
    del relu_1
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf466 = aten.convolution_backward(buf465, mean_1, primals_119, [25], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf465
    del mean_1
    del primals_119
    buf467 = buf466[0]
    buf468 = buf466[1]
    buf469 = buf466[2]
    del buf466
    buf470 = empty((300, ), device='cpu', dtype=torch.float32)
    buf471 = empty((300, ), device='cpu', dtype=torch.float32)
    buf472 = buf454; del buf454  # reuse
    buf473 = buf471; del buf471  # reuse
    buf474 = buf472; del buf472  # reuse
    cpp_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_60(c_void_p(buf474.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(add_75.data_ptr()), c_void_p(convolution_17.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(unsqueeze_814.data_ptr()), c_void_p(squeeze_43.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf470.data_ptr()))
    del add_75
    del buf467
    del convolution_15
    del convolution_17
    del primals_27
    del squeeze_43
    del unsqueeze_814
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf475 = aten.convolution_backward(buf474, mul_103, primals_118, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 300, [True, True, False])
    del buf474
    del mul_103
    del primals_118
    buf476 = buf475[0]
    buf477 = buf475[1]
    del buf475
    buf478 = empty((300, ), device='cpu', dtype=torch.float32)
    buf479 = empty((300, ), device='cpu', dtype=torch.float32)
    buf480 = empty((300, ), device='cpu', dtype=torch.float32)
    buf481 = buf476; del buf476  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_61(c_void_p(buf481.data_ptr()), c_void_p(mul_981.data_ptr()), c_void_p(convolution_14.data_ptr()), c_void_p(unsqueeze_826.data_ptr()), c_void_p(squeeze_40.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(buf478.data_ptr()), c_void_p(buf479.data_ptr()), c_void_p(buf480.data_ptr()))
    del buf479
    del convolution_14
    del mul_981
    del primals_25
    del squeeze_40
    del unsqueeze_826
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf482 = aten.convolution_backward(buf481, add_65, primals_117, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_65
    del buf481
    del primals_117
    buf483 = buf482[0]
    buf484 = buf482[1]
    del buf482
    buf485 = empty((50, ), device='cpu', dtype=torch.float32)
    buf486 = empty((50, ), device='cpu', dtype=torch.float32)
    buf487 = empty((50, ), device='cpu', dtype=torch.float32)
    buf488 = buf483; del buf483  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_62(c_void_p(buf488.data_ptr()), c_void_p(buf447.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(unsqueeze_838.data_ptr()), c_void_p(squeeze_37.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf485.data_ptr()), c_void_p(buf486.data_ptr()), c_void_p(buf487.data_ptr()))
    del buf447
    del buf486
    del convolution_13
    del primals_23
    del squeeze_37
    del unsqueeze_838
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf489 = aten.convolution_backward(buf488, clamp_max_3, primals_116, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf488
    del clamp_max_3
    del primals_116
    buf490 = buf489[0]
    buf491 = buf489[1]
    del buf489
    buf492 = empty_strided((8, 228, 1, 1), (228, 1, 1824, 1824), device='cpu', dtype=torch.float32)
    buf493 = reinterpret_tensor(buf492, (8, 228, 1, 1), (228, 1, 1, 1), 0); del buf492  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_63(c_void_p(buf493.data_ptr()), c_void_p(add_55.data_ptr()), c_void_p(convolution_12.data_ptr()), c_void_p(buf490.data_ptr()))
    # Source Nodes: [sigmoid], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf494 = aten.convolution_backward(buf493, relu, primals_114, [228], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf493
    del primals_114
    buf495 = buf494[0]
    buf496 = buf494[1]
    buf497 = buf494[2]
    del buf494
    buf498 = empty((19, ), device='cpu', dtype=torch.float32)
    buf499 = empty((19, ), device='cpu', dtype=torch.float32)
    buf500 = empty((19, ), device='cpu', dtype=torch.float32)
    buf501 = reinterpret_tensor(buf495, (8, 19, 1, 1), (19, 1, 1, 1), 0); del buf495  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_threshold_backward_64(c_void_p(buf501.data_ptr()), c_void_p(relu.data_ptr()), c_void_p(convolution_11.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(primals_112.data_ptr()), c_void_p(buf498.data_ptr()), c_void_p(buf499.data_ptr()), c_void_p(buf500.data_ptr()))
    del buf0
    del buf1
    del buf499
    del convolution_11
    del primals_112
    del relu
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf502 = aten.convolution_backward(buf501, mean, primals_110, [19], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf501
    del mean
    del primals_110
    buf503 = buf502[0]
    buf504 = buf502[1]
    buf505 = buf502[2]
    del buf502
    buf506 = empty((228, ), device='cpu', dtype=torch.float32)
    buf507 = empty((228, ), device='cpu', dtype=torch.float32)
    buf508 = buf490; del buf490  # reuse
    buf509 = buf507; del buf507  # reuse
    buf510 = buf508; del buf508  # reuse
    cpp_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_65(c_void_p(buf510.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(add_55.data_ptr()), c_void_p(convolution_12.data_ptr()), c_void_p(buf503.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(unsqueeze_862.data_ptr()), c_void_p(squeeze_31.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(buf506.data_ptr()))
    del add_55
    del buf503
    del convolution_10
    del convolution_12
    del primals_21
    del squeeze_31
    del unsqueeze_862
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf511 = aten.convolution_backward(buf510, mul_73, primals_109, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 228, [True, True, False])
    del buf510
    del mul_73
    del primals_109
    buf512 = buf511[0]
    buf513 = buf511[1]
    del buf511
    buf514 = empty((228, ), device='cpu', dtype=torch.float32)
    buf515 = empty((228, ), device='cpu', dtype=torch.float32)
    buf516 = empty((228, ), device='cpu', dtype=torch.float32)
    buf517 = buf512; del buf512  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_66(c_void_p(buf517.data_ptr()), c_void_p(mul_1024.data_ptr()), c_void_p(convolution_9.data_ptr()), c_void_p(unsqueeze_874.data_ptr()), c_void_p(squeeze_28.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf514.data_ptr()), c_void_p(buf515.data_ptr()), c_void_p(buf516.data_ptr()))
    del buf515
    del convolution_9
    del mul_1024
    del primals_19
    del squeeze_28
    del unsqueeze_874
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf518 = aten.convolution_backward(buf517, cat, primals_108, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf517
    del cat
    del primals_108
    buf519 = buf518[0]
    buf520 = buf518[1]
    del buf518
    buf521 = empty((38, ), device='cpu', dtype=torch.float32)
    buf522 = empty((38, ), device='cpu', dtype=torch.float32)
    buf523 = empty((38, ), device='cpu', dtype=torch.float32)
    buf524 = empty((8, 38, 56, 56), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_67(c_void_p(buf519.data_ptr()), c_void_p(convolution_8.data_ptr()), c_void_p(unsqueeze_886.data_ptr()), c_void_p(squeeze_25.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf521.data_ptr()), c_void_p(buf522.data_ptr()), c_void_p(buf523.data_ptr()), c_void_p(buf524.data_ptr()))
    del buf522
    del convolution_8
    del primals_17
    del squeeze_25
    del unsqueeze_886
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.slice_backward]
    buf525 = aten.convolution_backward(buf524, clamp_max_2, primals_107, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf524
    del clamp_max_2
    del primals_107
    buf526 = buf525[0]
    buf527 = buf525[1]
    del buf525
    buf528 = empty((162, ), device='cpu', dtype=torch.float32)
    buf529 = empty((162, ), device='cpu', dtype=torch.float32)
    buf530 = empty((162, ), device='cpu', dtype=torch.float32)
    buf531 = buf526; del buf526  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_68(c_void_p(buf531.data_ptr()), c_void_p(bitwise_or_13.data_ptr()), c_void_p(convolution_7.data_ptr()), c_void_p(unsqueeze_898.data_ptr()), c_void_p(squeeze_22.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf528.data_ptr()), c_void_p(buf529.data_ptr()), c_void_p(buf530.data_ptr()))
    del bitwise_or_13
    del convolution_7
    del primals_15
    del squeeze_22
    del unsqueeze_898
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
    buf532 = aten.convolution_backward(buf531, mul_51, primals_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 162, [True, True, False])
    del buf531
    del mul_51
    del primals_106
    buf533 = buf532[0]
    buf534 = buf532[1]
    del buf532
    buf535 = buf529; del buf529  # reuse
    buf536 = empty((162, ), device='cpu', dtype=torch.float32)
    buf537 = empty((162, ), device='cpu', dtype=torch.float32)
    buf538 = buf533; del buf533  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_69(c_void_p(buf538.data_ptr()), c_void_p(mul_1054.data_ptr()), c_void_p(convolution_6.data_ptr()), c_void_p(unsqueeze_910.data_ptr()), c_void_p(squeeze_19.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(buf535.data_ptr()), c_void_p(buf536.data_ptr()), c_void_p(buf537.data_ptr()))
    del buf536
    del convolution_6
    del mul_1054
    del primals_13
    del squeeze_19
    del unsqueeze_910
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf539 = aten.convolution_backward(buf538, add_29, primals_105, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_29
    del buf538
    del primals_105
    buf540 = buf539[0]
    buf541 = buf539[1]
    del buf539
    buf542 = empty((27, ), device='cpu', dtype=torch.float32)
    buf543 = empty((27, ), device='cpu', dtype=torch.float32)
    buf544 = empty((27, ), device='cpu', dtype=torch.float32)
    buf545 = buf540; del buf540  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_70(c_void_p(buf545.data_ptr()), c_void_p(buf519.data_ptr()), c_void_p(convolution_5.data_ptr()), c_void_p(unsqueeze_922.data_ptr()), c_void_p(squeeze_16.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf542.data_ptr()), c_void_p(buf543.data_ptr()), c_void_p(buf544.data_ptr()))
    del buf519
    del buf543
    del convolution_5
    del primals_11
    del squeeze_16
    del unsqueeze_922
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf546 = aten.convolution_backward(buf545, clamp_max_1, primals_104, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf545
    del clamp_max_1
    del primals_104
    buf547 = buf546[0]
    buf548 = buf546[1]
    del buf546
    buf549 = empty((96, ), device='cpu', dtype=torch.float32)
    buf550 = empty((96, ), device='cpu', dtype=torch.float32)
    buf551 = empty((96, ), device='cpu', dtype=torch.float32)
    buf552 = buf547; del buf547  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_71(c_void_p(buf552.data_ptr()), c_void_p(bitwise_or_14.data_ptr()), c_void_p(convolution_4.data_ptr()), c_void_p(unsqueeze_934.data_ptr()), c_void_p(squeeze_13.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf549.data_ptr()), c_void_p(buf550.data_ptr()), c_void_p(buf551.data_ptr()))
    del bitwise_or_14
    del convolution_4
    del primals_9
    del squeeze_13
    del unsqueeze_934
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
    buf553 = aten.convolution_backward(buf552, mul_29, primals_103, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 96, [True, True, False])
    del buf552
    del mul_29
    del primals_103
    buf554 = buf553[0]
    buf555 = buf553[1]
    del buf553
    buf556 = buf550; del buf550  # reuse
    buf557 = empty((96, ), device='cpu', dtype=torch.float32)
    buf558 = empty((96, ), device='cpu', dtype=torch.float32)
    buf559 = buf554; del buf554  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_72(c_void_p(buf559.data_ptr()), c_void_p(mul_1084.data_ptr()), c_void_p(convolution_3.data_ptr()), c_void_p(unsqueeze_946.data_ptr()), c_void_p(squeeze_10.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf556.data_ptr()), c_void_p(buf557.data_ptr()), c_void_p(buf558.data_ptr()))
    del buf557
    del convolution_3
    del mul_1084
    del primals_7
    del squeeze_10
    del unsqueeze_946
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf560 = aten.convolution_backward(buf559, add_14, primals_102, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_14
    del buf559
    del primals_102
    buf561 = buf560[0]
    buf562 = buf560[1]
    del buf560
    buf563 = empty((16, ), device='cpu', dtype=torch.float32)
    buf564 = empty((16, ), device='cpu', dtype=torch.float32)
    buf565 = empty((16, ), device='cpu', dtype=torch.float32)
    buf566 = buf561; del buf561  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_73(c_void_p(buf566.data_ptr()), c_void_p(convolution_2.data_ptr()), c_void_p(unsqueeze_958.data_ptr()), c_void_p(squeeze_7.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf563.data_ptr()), c_void_p(buf564.data_ptr()), c_void_p(buf565.data_ptr()))
    del buf564
    del convolution_2
    del primals_5
    del squeeze_7
    del unsqueeze_958
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf567 = aten.convolution_backward(buf566, clamp_max, primals_101, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf566
    del clamp_max
    del primals_101
    buf568 = buf567[0]
    buf569 = buf567[1]
    del buf567
    buf570 = empty((32, ), device='cpu', dtype=torch.float32)
    buf571 = empty((32, ), device='cpu', dtype=torch.float32)
    buf572 = empty((32, ), device='cpu', dtype=torch.float32)
    buf573 = buf568; del buf568  # reuse
    cpp_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_74(c_void_p(buf573.data_ptr()), c_void_p(bitwise_or_15.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(unsqueeze_970.data_ptr()), c_void_p(squeeze_4.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf570.data_ptr()), c_void_p(buf571.data_ptr()), c_void_p(buf572.data_ptr()))
    del bitwise_or_15
    del convolution_1
    del primals_3
    del squeeze_4
    del unsqueeze_970
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
    buf574 = aten.convolution_backward(buf573, mul_7, primals_100, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
    del buf573
    del mul_7
    del primals_100
    buf575 = buf574[0]
    buf576 = buf574[1]
    del buf574
    buf577 = buf571; del buf571  # reuse
    buf578 = empty((32, ), device='cpu', dtype=torch.float32)
    buf579 = empty((32, ), device='cpu', dtype=torch.float32)
    buf580 = buf575; del buf575  # reuse
    cpp_fused_convolution_backward_mul_native_batch_norm_backward_75(c_void_p(buf580.data_ptr()), c_void_p(mul_1114.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(unsqueeze_982.data_ptr()), c_void_p(squeeze_1.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf577.data_ptr()), c_void_p(buf578.data_ptr()), c_void_p(buf579.data_ptr()))
    del buf578
    del convolution
    del mul_1114
    del primals_1
    del squeeze_1
    del unsqueeze_982
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
    buf581 = aten.convolution_backward(buf580, primals_414, primals_99, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
    del buf580
    del primals_414
    del primals_99
    buf582 = buf581[1]
    return (buf579, buf577, buf572, buf570, buf565, buf563, buf558, buf556, buf551, buf549, buf544, buf542, buf537, buf535, buf530, buf528, buf523, buf521, buf516, buf514, buf509, buf506, buf487, buf485, buf480, buf478, buf473, buf470, buf451, buf449, buf444, buf442, buf437, buf434, buf415, buf413, buf408, buf406, buf401, buf398, buf379, buf377, buf371, buf369, buf364, buf361, buf342, buf339, buf334, buf332, buf327, buf324, buf306, buf303, buf298, buf296, buf291, buf288, buf269, buf267, buf262, buf260, buf255, buf252, buf233, buf231, buf226, buf224, buf219, buf216, buf197, buf195, buf189, buf187, buf182, buf179, buf160, buf157, buf152, buf150, buf145, buf142, buf124, buf121, buf116, buf114, buf109, buf106, buf87, buf85, buf80, buf78, buf73, buf70, buf51, buf49, buf44, buf42, buf582, buf576, buf569, buf562, buf555, buf548, buf541, buf534, buf527, buf520, buf513, buf504, buf505, buf500, buf498, buf496, buf497, buf491, buf484, buf477, buf468, buf469, buf464, buf462, buf460, buf461, buf455, buf448, buf441, buf432, buf433, buf428, buf426, buf424, buf425, buf419, buf412, buf405, buf396, buf397, buf392, buf390, buf388, buf389, buf383, buf375, buf368, buf359, buf360, buf355, buf353, buf351, buf352, buf346, buf338, buf331, buf322, buf323, buf318, buf316, buf314, buf315, buf309, buf302, buf295, buf286, buf287, buf282, buf280, buf278, buf279, buf273, buf266, buf259, buf250, buf251, buf246, buf244, buf242, buf243, buf237, buf230, buf223, buf214, buf215, buf210, buf208, buf206, buf207, buf201, buf193, buf186, buf177, buf178, buf173, buf171, buf169, buf170, buf164, buf156, buf149, buf140, buf141, buf136, buf134, buf132, buf133, buf127, buf120, buf113, buf104, buf105, buf100, buf98, buf96, buf97, buf91, buf84, buf77, buf68, buf69, buf64, buf62, buf60, buf61, buf55, buf48, reinterpret_tensor(buf40, (1000, 1280), (1280, 1), 0), reinterpret_tensor(buf41, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((27, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((162, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((162, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((38, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((228, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((228, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((50, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((300, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((300, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((61, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((366, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((366, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((432, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((432, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((84, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((504, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((504, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((95, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((570, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((570, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((106, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((636, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((636, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((117, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((702, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((702, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((140, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((840, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((840, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((151, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((906, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((906, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((162, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((972, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((972, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((174, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((1044, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((1044, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((185, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
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
    primals_112 = rand_strided((19, ), (1, ), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((228, 19, 1, 1), (19, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((50, 228, 1, 1), (228, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((300, 50, 1, 1), (50, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((300, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((25, 300, 1, 1), (300, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((25, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((300, 25, 1, 1), (25, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((61, 300, 1, 1), (300, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((366, 61, 1, 1), (61, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((366, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((30, 366, 1, 1), (366, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((30, ), (1, ), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((366, 30, 1, 1), (30, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((72, 366, 1, 1), (366, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((432, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((36, 432, 1, 1), (432, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((432, 36, 1, 1), (36, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((84, 432, 1, 1), (432, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((504, 84, 1, 1), (84, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((504, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((42, 504, 1, 1), (504, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((42, ), (1, ), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((504, 42, 1, 1), (42, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((95, 504, 1, 1), (504, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((570, 95, 1, 1), (95, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((570, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((47, 570, 1, 1), (570, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((47, ), (1, ), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((570, 47, 1, 1), (47, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((106, 570, 1, 1), (570, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((636, 106, 1, 1), (106, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((636, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((53, 636, 1, 1), (636, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((53, ), (1, ), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((636, 53, 1, 1), (53, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((117, 636, 1, 1), (636, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((702, 117, 1, 1), (117, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((702, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((58, 702, 1, 1), (702, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((58, ), (1, ), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((702, 58, 1, 1), (58, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((128, 702, 1, 1), (702, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((768, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((64, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((768, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((140, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((840, 140, 1, 1), (140, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((840, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((70, 840, 1, 1), (840, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((70, ), (1, ), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((840, 70, 1, 1), (70, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((151, 840, 1, 1), (840, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((906, 151, 1, 1), (151, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((906, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((75, 906, 1, 1), (906, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((75, ), (1, ), device='cpu', dtype=torch.float32)
    primals_204 = rand_strided((906, 75, 1, 1), (75, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_206 = rand_strided((162, 906, 1, 1), (906, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_207 = rand_strided((972, 162, 1, 1), (162, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((972, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_209 = rand_strided((81, 972, 1, 1), (972, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((81, ), (1, ), device='cpu', dtype=torch.float32)
    primals_213 = rand_strided((972, 81, 1, 1), (81, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_215 = rand_strided((174, 972, 1, 1), (972, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_216 = rand_strided((1044, 174, 1, 1), (174, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((1044, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_218 = rand_strided((87, 1044, 1, 1), (1044, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_220 = rand_strided((87, ), (1, ), device='cpu', dtype=torch.float32)
    primals_222 = rand_strided((1044, 87, 1, 1), (87, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_224 = rand_strided((185, 1044, 1, 1), (1044, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_225 = rand_strided((1280, 185, 1, 1), (185, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_414 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    convolution = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    squeeze_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    mul_7 = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    convolution_1 = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    squeeze_4 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    clamp_max = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    squeeze_7 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    add_14 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    convolution_3 = rand_strided((8, 96, 112, 112), (1204224, 1, 10752, 96), device='cpu', dtype=torch.float32)
    squeeze_10 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    mul_29 = rand_strided((8, 96, 112, 112), (1204224, 1, 10752, 96), device='cpu', dtype=torch.float32)
    convolution_4 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cpu', dtype=torch.float32)
    squeeze_13 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    clamp_max_1 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cpu', dtype=torch.float32)
    convolution_5 = rand_strided((8, 27, 56, 56), (84672, 1, 1512, 27), device='cpu', dtype=torch.float32)
    squeeze_16 = rand_strided((27, ), (1, ), device='cpu', dtype=torch.float32)
    add_29 = rand_strided((8, 27, 56, 56), (84672, 1, 1512, 27), device='cpu', dtype=torch.float32)
    convolution_6 = rand_strided((8, 162, 56, 56), (508032, 1, 9072, 162), device='cpu', dtype=torch.float32)
    squeeze_19 = rand_strided((162, ), (1, ), device='cpu', dtype=torch.float32)
    mul_51 = rand_strided((8, 162, 56, 56), (508032, 1, 9072, 162), device='cpu', dtype=torch.float32)
    convolution_7 = rand_strided((8, 162, 56, 56), (508032, 1, 9072, 162), device='cpu', dtype=torch.float32)
    squeeze_22 = rand_strided((162, ), (1, ), device='cpu', dtype=torch.float32)
    clamp_max_2 = rand_strided((8, 162, 56, 56), (508032, 1, 9072, 162), device='cpu', dtype=torch.float32)
    convolution_8 = rand_strided((8, 38, 56, 56), (119168, 1, 2128, 38), device='cpu', dtype=torch.float32)
    squeeze_25 = rand_strided((38, ), (1, ), device='cpu', dtype=torch.float32)
    cat = rand_strided((8, 38, 56, 56), (119168, 1, 2128, 38), device='cpu', dtype=torch.float32)
    convolution_9 = rand_strided((8, 228, 56, 56), (715008, 1, 12768, 228), device='cpu', dtype=torch.float32)
    squeeze_28 = rand_strided((228, ), (1, ), device='cpu', dtype=torch.float32)
    mul_73 = rand_strided((8, 228, 56, 56), (715008, 1, 12768, 228), device='cpu', dtype=torch.float32)
    convolution_10 = rand_strided((8, 228, 28, 28), (178752, 1, 6384, 228), device='cpu', dtype=torch.float32)
    squeeze_31 = rand_strided((228, ), (1, ), device='cpu', dtype=torch.float32)
    add_55 = rand_strided((8, 228, 28, 28), (178752, 1, 6384, 228), device='cpu', dtype=torch.float32)
    mean = rand_strided((8, 228, 1, 1), (228, 1, 228, 228), device='cpu', dtype=torch.float32)
    convolution_11 = rand_strided((8, 19, 1, 1), (19, 1, 19, 19), device='cpu', dtype=torch.float32)
    relu = rand_strided((8, 19, 1, 1), (19, 1, 19, 19), device='cpu', dtype=torch.float32)
    convolution_12 = rand_strided((8, 228, 1, 1), (228, 1, 228, 228), device='cpu', dtype=torch.float32)
    clamp_max_3 = rand_strided((8, 228, 28, 28), (178752, 1, 6384, 228), device='cpu', dtype=torch.float32)
    convolution_13 = rand_strided((8, 50, 28, 28), (39200, 1, 1400, 50), device='cpu', dtype=torch.float32)
    squeeze_37 = rand_strided((50, ), (1, ), device='cpu', dtype=torch.float32)
    add_65 = rand_strided((8, 50, 28, 28), (39200, 1, 1400, 50), device='cpu', dtype=torch.float32)
    convolution_14 = rand_strided((8, 300, 28, 28), (235200, 1, 8400, 300), device='cpu', dtype=torch.float32)
    squeeze_40 = rand_strided((300, ), (1, ), device='cpu', dtype=torch.float32)
    mul_103 = rand_strided((8, 300, 28, 28), (235200, 1, 8400, 300), device='cpu', dtype=torch.float32)
    convolution_15 = rand_strided((8, 300, 28, 28), (235200, 1, 8400, 300), device='cpu', dtype=torch.float32)
    squeeze_43 = rand_strided((300, ), (1, ), device='cpu', dtype=torch.float32)
    add_75 = rand_strided((8, 300, 28, 28), (235200, 1, 8400, 300), device='cpu', dtype=torch.float32)
    mean_1 = rand_strided((8, 300, 1, 1), (300, 1, 300, 300), device='cpu', dtype=torch.float32)
    convolution_16 = rand_strided((8, 25, 1, 1), (25, 1, 25, 25), device='cpu', dtype=torch.float32)
    relu_1 = rand_strided((8, 25, 1, 1), (25, 1, 25, 25), device='cpu', dtype=torch.float32)
    convolution_17 = rand_strided((8, 300, 1, 1), (300, 1, 300, 300), device='cpu', dtype=torch.float32)
    clamp_max_4 = rand_strided((8, 300, 28, 28), (235200, 1, 8400, 300), device='cpu', dtype=torch.float32)
    convolution_18 = rand_strided((8, 61, 28, 28), (47824, 1, 1708, 61), device='cpu', dtype=torch.float32)
    squeeze_49 = rand_strided((61, ), (1, ), device='cpu', dtype=torch.float32)
    cat_1 = rand_strided((8, 61, 28, 28), (47824, 1, 1708, 61), device='cpu', dtype=torch.float32)
    convolution_19 = rand_strided((8, 366, 28, 28), (286944, 1, 10248, 366), device='cpu', dtype=torch.float32)
    squeeze_52 = rand_strided((366, ), (1, ), device='cpu', dtype=torch.float32)
    mul_133 = rand_strided((8, 366, 28, 28), (286944, 1, 10248, 366), device='cpu', dtype=torch.float32)
    convolution_20 = rand_strided((8, 366, 14, 14), (71736, 1, 5124, 366), device='cpu', dtype=torch.float32)
    squeeze_55 = rand_strided((366, ), (1, ), device='cpu', dtype=torch.float32)
    add_96 = rand_strided((8, 366, 14, 14), (71736, 1, 5124, 366), device='cpu', dtype=torch.float32)
    mean_2 = rand_strided((8, 366, 1, 1), (366, 1, 366, 366), device='cpu', dtype=torch.float32)
    convolution_21 = rand_strided((8, 30, 1, 1), (30, 1, 30, 30), device='cpu', dtype=torch.float32)
    relu_2 = rand_strided((8, 30, 1, 1), (30, 1, 30, 30), device='cpu', dtype=torch.float32)
    convolution_22 = rand_strided((8, 366, 1, 1), (366, 1, 366, 366), device='cpu', dtype=torch.float32)
    clamp_max_5 = rand_strided((8, 366, 14, 14), (71736, 1, 5124, 366), device='cpu', dtype=torch.float32)
    convolution_23 = rand_strided((8, 72, 14, 14), (14112, 1, 1008, 72), device='cpu', dtype=torch.float32)
    squeeze_61 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    add_106 = rand_strided((8, 72, 14, 14), (14112, 1, 1008, 72), device='cpu', dtype=torch.float32)
    convolution_24 = rand_strided((8, 432, 14, 14), (84672, 1, 6048, 432), device='cpu', dtype=torch.float32)
    squeeze_64 = rand_strided((432, ), (1, ), device='cpu', dtype=torch.float32)
    mul_163 = rand_strided((8, 432, 14, 14), (84672, 1, 6048, 432), device='cpu', dtype=torch.float32)
    convolution_25 = rand_strided((8, 432, 14, 14), (84672, 1, 6048, 432), device='cpu', dtype=torch.float32)
    squeeze_67 = rand_strided((432, ), (1, ), device='cpu', dtype=torch.float32)
    add_116 = rand_strided((8, 432, 14, 14), (84672, 1, 6048, 432), device='cpu', dtype=torch.float32)
    mean_3 = rand_strided((8, 432, 1, 1), (432, 1, 432, 432), device='cpu', dtype=torch.float32)
    convolution_26 = rand_strided((8, 36, 1, 1), (36, 1, 36, 36), device='cpu', dtype=torch.float32)
    relu_3 = rand_strided((8, 36, 1, 1), (36, 1, 36, 36), device='cpu', dtype=torch.float32)
    convolution_27 = rand_strided((8, 432, 1, 1), (432, 1, 432, 432), device='cpu', dtype=torch.float32)
    clamp_max_6 = rand_strided((8, 432, 14, 14), (84672, 1, 6048, 432), device='cpu', dtype=torch.float32)
    convolution_28 = rand_strided((8, 84, 14, 14), (16464, 1, 1176, 84), device='cpu', dtype=torch.float32)
    squeeze_73 = rand_strided((84, ), (1, ), device='cpu', dtype=torch.float32)
    cat_2 = rand_strided((8, 84, 14, 14), (16464, 1, 1176, 84), device='cpu', dtype=torch.float32)
    convolution_29 = rand_strided((8, 504, 14, 14), (98784, 1, 7056, 504), device='cpu', dtype=torch.float32)
    squeeze_76 = rand_strided((504, ), (1, ), device='cpu', dtype=torch.float32)
    mul_193 = rand_strided((8, 504, 14, 14), (98784, 1, 7056, 504), device='cpu', dtype=torch.float32)
    convolution_30 = rand_strided((8, 504, 14, 14), (98784, 1, 7056, 504), device='cpu', dtype=torch.float32)
    squeeze_79 = rand_strided((504, ), (1, ), device='cpu', dtype=torch.float32)
    add_137 = rand_strided((8, 504, 14, 14), (98784, 1, 7056, 504), device='cpu', dtype=torch.float32)
    mean_4 = rand_strided((8, 504, 1, 1), (504, 1, 504, 504), device='cpu', dtype=torch.float32)
    convolution_31 = rand_strided((8, 42, 1, 1), (42, 1, 42, 42), device='cpu', dtype=torch.float32)
    relu_4 = rand_strided((8, 42, 1, 1), (42, 1, 42, 42), device='cpu', dtype=torch.float32)
    convolution_32 = rand_strided((8, 504, 1, 1), (504, 1, 504, 504), device='cpu', dtype=torch.float32)
    clamp_max_7 = rand_strided((8, 504, 14, 14), (98784, 1, 7056, 504), device='cpu', dtype=torch.float32)
    convolution_33 = rand_strided((8, 95, 14, 14), (18620, 1, 1330, 95), device='cpu', dtype=torch.float32)
    squeeze_85 = rand_strided((95, ), (1, ), device='cpu', dtype=torch.float32)
    cat_3 = rand_strided((8, 95, 14, 14), (18620, 1, 1330, 95), device='cpu', dtype=torch.float32)
    convolution_34 = rand_strided((8, 570, 14, 14), (111720, 1, 7980, 570), device='cpu', dtype=torch.float32)
    squeeze_88 = rand_strided((570, ), (1, ), device='cpu', dtype=torch.float32)
    mul_223 = rand_strided((8, 570, 14, 14), (111720, 1, 7980, 570), device='cpu', dtype=torch.float32)
    convolution_35 = rand_strided((8, 570, 14, 14), (111720, 1, 7980, 570), device='cpu', dtype=torch.float32)
    squeeze_91 = rand_strided((570, ), (1, ), device='cpu', dtype=torch.float32)
    add_158 = rand_strided((8, 570, 14, 14), (111720, 1, 7980, 570), device='cpu', dtype=torch.float32)
    mean_5 = rand_strided((8, 570, 1, 1), (570, 1, 570, 570), device='cpu', dtype=torch.float32)
    convolution_36 = rand_strided((8, 47, 1, 1), (47, 1, 47, 47), device='cpu', dtype=torch.float32)
    relu_5 = rand_strided((8, 47, 1, 1), (47, 1, 47, 47), device='cpu', dtype=torch.float32)
    convolution_37 = rand_strided((8, 570, 1, 1), (570, 1, 570, 570), device='cpu', dtype=torch.float32)
    clamp_max_8 = rand_strided((8, 570, 14, 14), (111720, 1, 7980, 570), device='cpu', dtype=torch.float32)
    convolution_38 = rand_strided((8, 106, 14, 14), (20776, 1, 1484, 106), device='cpu', dtype=torch.float32)
    squeeze_97 = rand_strided((106, ), (1, ), device='cpu', dtype=torch.float32)
    cat_4 = rand_strided((8, 106, 14, 14), (20776, 1, 1484, 106), device='cpu', dtype=torch.float32)
    convolution_39 = rand_strided((8, 636, 14, 14), (124656, 1, 8904, 636), device='cpu', dtype=torch.float32)
    squeeze_100 = rand_strided((636, ), (1, ), device='cpu', dtype=torch.float32)
    mul_253 = rand_strided((8, 636, 14, 14), (124656, 1, 8904, 636), device='cpu', dtype=torch.float32)
    convolution_40 = rand_strided((8, 636, 14, 14), (124656, 1, 8904, 636), device='cpu', dtype=torch.float32)
    squeeze_103 = rand_strided((636, ), (1, ), device='cpu', dtype=torch.float32)
    add_179 = rand_strided((8, 636, 14, 14), (124656, 1, 8904, 636), device='cpu', dtype=torch.float32)
    mean_6 = rand_strided((8, 636, 1, 1), (636, 1, 636, 636), device='cpu', dtype=torch.float32)
    convolution_41 = rand_strided((8, 53, 1, 1), (53, 1, 53, 53), device='cpu', dtype=torch.float32)
    relu_6 = rand_strided((8, 53, 1, 1), (53, 1, 53, 53), device='cpu', dtype=torch.float32)
    convolution_42 = rand_strided((8, 636, 1, 1), (636, 1, 636, 636), device='cpu', dtype=torch.float32)
    clamp_max_9 = rand_strided((8, 636, 14, 14), (124656, 1, 8904, 636), device='cpu', dtype=torch.float32)
    convolution_43 = rand_strided((8, 117, 14, 14), (22932, 1, 1638, 117), device='cpu', dtype=torch.float32)
    squeeze_109 = rand_strided((117, ), (1, ), device='cpu', dtype=torch.float32)
    cat_5 = rand_strided((8, 117, 14, 14), (22932, 1, 1638, 117), device='cpu', dtype=torch.float32)
    convolution_44 = rand_strided((8, 702, 14, 14), (137592, 1, 9828, 702), device='cpu', dtype=torch.float32)
    squeeze_112 = rand_strided((702, ), (1, ), device='cpu', dtype=torch.float32)
    mul_283 = rand_strided((8, 702, 14, 14), (137592, 1, 9828, 702), device='cpu', dtype=torch.float32)
    convolution_45 = rand_strided((8, 702, 14, 14), (137592, 1, 9828, 702), device='cpu', dtype=torch.float32)
    squeeze_115 = rand_strided((702, ), (1, ), device='cpu', dtype=torch.float32)
    add_200 = rand_strided((8, 702, 14, 14), (137592, 1, 9828, 702), device='cpu', dtype=torch.float32)
    mean_7 = rand_strided((8, 702, 1, 1), (702, 1, 702, 702), device='cpu', dtype=torch.float32)
    convolution_46 = rand_strided((8, 58, 1, 1), (58, 1, 58, 58), device='cpu', dtype=torch.float32)
    relu_7 = rand_strided((8, 58, 1, 1), (58, 1, 58, 58), device='cpu', dtype=torch.float32)
    convolution_47 = rand_strided((8, 702, 1, 1), (702, 1, 702, 702), device='cpu', dtype=torch.float32)
    clamp_max_10 = rand_strided((8, 702, 14, 14), (137592, 1, 9828, 702), device='cpu', dtype=torch.float32)
    convolution_48 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    squeeze_121 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    cat_6 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    convolution_49 = rand_strided((8, 768, 14, 14), (150528, 1, 10752, 768), device='cpu', dtype=torch.float32)
    squeeze_124 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    mul_313 = rand_strided((8, 768, 14, 14), (150528, 1, 10752, 768), device='cpu', dtype=torch.float32)
    convolution_50 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    squeeze_127 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_221 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    mean_8 = rand_strided((8, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    convolution_51 = rand_strided((8, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    relu_8 = rand_strided((8, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    convolution_52 = rand_strided((8, 768, 1, 1), (768, 1, 768, 768), device='cpu', dtype=torch.float32)
    clamp_max_11 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cpu', dtype=torch.float32)
    convolution_53 = rand_strided((8, 140, 7, 7), (6860, 1, 980, 140), device='cpu', dtype=torch.float32)
    squeeze_133 = rand_strided((140, ), (1, ), device='cpu', dtype=torch.float32)
    add_231 = rand_strided((8, 140, 7, 7), (6860, 1, 980, 140), device='cpu', dtype=torch.float32)
    convolution_54 = rand_strided((8, 840, 7, 7), (41160, 1, 5880, 840), device='cpu', dtype=torch.float32)
    squeeze_136 = rand_strided((840, ), (1, ), device='cpu', dtype=torch.float32)
    mul_343 = rand_strided((8, 840, 7, 7), (41160, 1, 5880, 840), device='cpu', dtype=torch.float32)
    convolution_55 = rand_strided((8, 840, 7, 7), (41160, 1, 5880, 840), device='cpu', dtype=torch.float32)
    squeeze_139 = rand_strided((840, ), (1, ), device='cpu', dtype=torch.float32)
    add_241 = rand_strided((8, 840, 7, 7), (41160, 1, 5880, 840), device='cpu', dtype=torch.float32)
    mean_9 = rand_strided((8, 840, 1, 1), (840, 1, 840, 840), device='cpu', dtype=torch.float32)
    convolution_56 = rand_strided((8, 70, 1, 1), (70, 1, 70, 70), device='cpu', dtype=torch.float32)
    relu_9 = rand_strided((8, 70, 1, 1), (70, 1, 70, 70), device='cpu', dtype=torch.float32)
    convolution_57 = rand_strided((8, 840, 1, 1), (840, 1, 840, 840), device='cpu', dtype=torch.float32)
    clamp_max_12 = rand_strided((8, 840, 7, 7), (41160, 1, 5880, 840), device='cpu', dtype=torch.float32)
    convolution_58 = rand_strided((8, 151, 7, 7), (7399, 1, 1057, 151), device='cpu', dtype=torch.float32)
    squeeze_145 = rand_strided((151, ), (1, ), device='cpu', dtype=torch.float32)
    cat_7 = rand_strided((8, 151, 7, 7), (7399, 1, 1057, 151), device='cpu', dtype=torch.float32)
    convolution_59 = rand_strided((8, 906, 7, 7), (44394, 1, 6342, 906), device='cpu', dtype=torch.float32)
    squeeze_148 = rand_strided((906, ), (1, ), device='cpu', dtype=torch.float32)
    mul_373 = rand_strided((8, 906, 7, 7), (44394, 1, 6342, 906), device='cpu', dtype=torch.float32)
    convolution_60 = rand_strided((8, 906, 7, 7), (44394, 1, 6342, 906), device='cpu', dtype=torch.float32)
    squeeze_151 = rand_strided((906, ), (1, ), device='cpu', dtype=torch.float32)
    add_262 = rand_strided((8, 906, 7, 7), (44394, 1, 6342, 906), device='cpu', dtype=torch.float32)
    mean_10 = rand_strided((8, 906, 1, 1), (906, 1, 906, 906), device='cpu', dtype=torch.float32)
    convolution_61 = rand_strided((8, 75, 1, 1), (75, 1, 75, 75), device='cpu', dtype=torch.float32)
    relu_10 = rand_strided((8, 75, 1, 1), (75, 1, 75, 75), device='cpu', dtype=torch.float32)
    convolution_62 = rand_strided((8, 906, 1, 1), (906, 1, 906, 906), device='cpu', dtype=torch.float32)
    clamp_max_13 = rand_strided((8, 906, 7, 7), (44394, 1, 6342, 906), device='cpu', dtype=torch.float32)
    convolution_63 = rand_strided((8, 162, 7, 7), (7938, 1, 1134, 162), device='cpu', dtype=torch.float32)
    squeeze_157 = rand_strided((162, ), (1, ), device='cpu', dtype=torch.float32)
    cat_8 = rand_strided((8, 162, 7, 7), (7938, 1, 1134, 162), device='cpu', dtype=torch.float32)
    convolution_64 = rand_strided((8, 972, 7, 7), (47628, 1, 6804, 972), device='cpu', dtype=torch.float32)
    squeeze_160 = rand_strided((972, ), (1, ), device='cpu', dtype=torch.float32)
    mul_403 = rand_strided((8, 972, 7, 7), (47628, 1, 6804, 972), device='cpu', dtype=torch.float32)
    convolution_65 = rand_strided((8, 972, 7, 7), (47628, 1, 6804, 972), device='cpu', dtype=torch.float32)
    squeeze_163 = rand_strided((972, ), (1, ), device='cpu', dtype=torch.float32)
    add_283 = rand_strided((8, 972, 7, 7), (47628, 1, 6804, 972), device='cpu', dtype=torch.float32)
    mean_11 = rand_strided((8, 972, 1, 1), (972, 1, 972, 972), device='cpu', dtype=torch.float32)
    convolution_66 = rand_strided((8, 81, 1, 1), (81, 1, 81, 81), device='cpu', dtype=torch.float32)
    relu_11 = rand_strided((8, 81, 1, 1), (81, 1, 81, 81), device='cpu', dtype=torch.float32)
    convolution_67 = rand_strided((8, 972, 1, 1), (972, 1, 972, 972), device='cpu', dtype=torch.float32)
    clamp_max_14 = rand_strided((8, 972, 7, 7), (47628, 1, 6804, 972), device='cpu', dtype=torch.float32)
    convolution_68 = rand_strided((8, 174, 7, 7), (8526, 1, 1218, 174), device='cpu', dtype=torch.float32)
    squeeze_169 = rand_strided((174, ), (1, ), device='cpu', dtype=torch.float32)
    cat_9 = rand_strided((8, 174, 7, 7), (8526, 1, 1218, 174), device='cpu', dtype=torch.float32)
    convolution_69 = rand_strided((8, 1044, 7, 7), (51156, 1, 7308, 1044), device='cpu', dtype=torch.float32)
    squeeze_172 = rand_strided((1044, ), (1, ), device='cpu', dtype=torch.float32)
    mul_433 = rand_strided((8, 1044, 7, 7), (51156, 1, 7308, 1044), device='cpu', dtype=torch.float32)
    convolution_70 = rand_strided((8, 1044, 7, 7), (51156, 1, 7308, 1044), device='cpu', dtype=torch.float32)
    squeeze_175 = rand_strided((1044, ), (1, ), device='cpu', dtype=torch.float32)
    add_304 = rand_strided((8, 1044, 7, 7), (51156, 1, 7308, 1044), device='cpu', dtype=torch.float32)
    mean_12 = rand_strided((8, 1044, 1, 1), (1044, 1, 1044, 1044), device='cpu', dtype=torch.float32)
    convolution_71 = rand_strided((8, 87, 1, 1), (87, 1, 87, 87), device='cpu', dtype=torch.float32)
    relu_12 = rand_strided((8, 87, 1, 1), (87, 1, 87, 87), device='cpu', dtype=torch.float32)
    convolution_72 = rand_strided((8, 1044, 1, 1), (1044, 1, 1044, 1044), device='cpu', dtype=torch.float32)
    clamp_max_15 = rand_strided((8, 1044, 7, 7), (51156, 1, 7308, 1044), device='cpu', dtype=torch.float32)
    convolution_73 = rand_strided((8, 185, 7, 7), (9065, 1, 1295, 185), device='cpu', dtype=torch.float32)
    squeeze_181 = rand_strided((185, ), (1, ), device='cpu', dtype=torch.float32)
    cat_10 = rand_strided((8, 185, 7, 7), (9065, 1, 1295, 185), device='cpu', dtype=torch.float32)
    convolution_74 = rand_strided((8, 1280, 7, 7), (62720, 1, 8960, 1280), device='cpu', dtype=torch.float32)
    squeeze_184 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    clone_17 = rand_strided((8, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    permute_1 = rand_strided((1000, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    mul_465 = rand_strided((8, 1280, 7, 7), (62720, 1, 8960, 1280), device='cpu', dtype=torch.float32)
    unsqueeze_250 = rand_strided((1, 1280, 1, 1), (1280, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_262 = rand_strided((1, 185, 1, 1), (185, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_286 = rand_strided((1, 1044, 1, 1), (1044, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_508 = rand_strided((8, 1044, 7, 7), (51156, 1, 7308, 1044), device='cpu', dtype=torch.float32)
    unsqueeze_298 = rand_strided((1, 1044, 1, 1), (1044, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_310 = rand_strided((1, 174, 1, 1), (174, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_334 = rand_strided((1, 972, 1, 1), (972, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_551 = rand_strided((8, 972, 7, 7), (47628, 1, 6804, 972), device='cpu', dtype=torch.float32)
    unsqueeze_346 = rand_strided((1, 972, 1, 1), (972, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_358 = rand_strided((1, 162, 1, 1), (162, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_382 = rand_strided((1, 906, 1, 1), (906, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_594 = rand_strided((8, 906, 7, 7), (44394, 1, 6342, 906), device='cpu', dtype=torch.float32)
    unsqueeze_394 = rand_strided((1, 906, 1, 1), (906, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_406 = rand_strided((1, 151, 1, 1), (151, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_430 = rand_strided((1, 840, 1, 1), (840, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_637 = rand_strided((8, 840, 7, 7), (41160, 1, 5880, 840), device='cpu', dtype=torch.float32)
    unsqueeze_442 = rand_strided((1, 840, 1, 1), (840, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_454 = rand_strided((1, 140, 1, 1), (140, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_478 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_680 = rand_strided((8, 768, 14, 14), (150528, 1, 10752, 768), device='cpu', dtype=torch.float32)
    unsqueeze_490 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_502 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_526 = rand_strided((1, 702, 1, 1), (702, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_723 = rand_strided((8, 702, 14, 14), (137592, 1, 9828, 702), device='cpu', dtype=torch.float32)
    unsqueeze_538 = rand_strided((1, 702, 1, 1), (702, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_550 = rand_strided((1, 117, 1, 1), (117, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_574 = rand_strided((1, 636, 1, 1), (636, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_766 = rand_strided((8, 636, 14, 14), (124656, 1, 8904, 636), device='cpu', dtype=torch.float32)
    unsqueeze_586 = rand_strided((1, 636, 1, 1), (636, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_598 = rand_strided((1, 106, 1, 1), (106, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_622 = rand_strided((1, 570, 1, 1), (570, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_809 = rand_strided((8, 570, 14, 14), (111720, 1, 7980, 570), device='cpu', dtype=torch.float32)
    unsqueeze_634 = rand_strided((1, 570, 1, 1), (570, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_646 = rand_strided((1, 95, 1, 1), (95, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_670 = rand_strided((1, 504, 1, 1), (504, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_852 = rand_strided((8, 504, 14, 14), (98784, 1, 7056, 504), device='cpu', dtype=torch.float32)
    unsqueeze_682 = rand_strided((1, 504, 1, 1), (504, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_694 = rand_strided((1, 84, 1, 1), (84, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_718 = rand_strided((1, 432, 1, 1), (432, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_895 = rand_strided((8, 432, 14, 14), (84672, 1, 6048, 432), device='cpu', dtype=torch.float32)
    unsqueeze_730 = rand_strided((1, 432, 1, 1), (432, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_742 = rand_strided((1, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_766 = rand_strided((1, 366, 1, 1), (366, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_938 = rand_strided((8, 366, 28, 28), (286944, 1, 10248, 366), device='cpu', dtype=torch.float32)
    unsqueeze_778 = rand_strided((1, 366, 1, 1), (366, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_790 = rand_strided((1, 61, 1, 1), (61, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_814 = rand_strided((1, 300, 1, 1), (300, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_981 = rand_strided((8, 300, 28, 28), (235200, 1, 8400, 300), device='cpu', dtype=torch.float32)
    unsqueeze_826 = rand_strided((1, 300, 1, 1), (300, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_838 = rand_strided((1, 50, 1, 1), (50, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_862 = rand_strided((1, 228, 1, 1), (228, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_1024 = rand_strided((8, 228, 56, 56), (715008, 1, 12768, 228), device='cpu', dtype=torch.float32)
    unsqueeze_874 = rand_strided((1, 228, 1, 1), (228, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_886 = rand_strided((1, 38, 1, 1), (38, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_or_13 = rand_strided((8, 162, 56, 56), (508032, 1, 9072, 162), device='cpu', dtype=torch.bool)
    unsqueeze_898 = rand_strided((1, 162, 1, 1), (162, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_1054 = rand_strided((8, 162, 56, 56), (508032, 1, 9072, 162), device='cpu', dtype=torch.float32)
    unsqueeze_910 = rand_strided((1, 162, 1, 1), (162, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_922 = rand_strided((1, 27, 1, 1), (27, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_or_14 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cpu', dtype=torch.bool)
    unsqueeze_934 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_1084 = rand_strided((8, 96, 112, 112), (1204224, 1, 10752, 96), device='cpu', dtype=torch.float32)
    unsqueeze_946 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_958 = rand_strided((1, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    bitwise_or_15 = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.bool)
    unsqueeze_970 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    mul_1114 = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    unsqueeze_982 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_112, primals_114, primals_116, primals_117, primals_118, primals_119, primals_121, primals_123, primals_125, primals_126, primals_127, primals_128, primals_130, primals_132, primals_134, primals_135, primals_136, primals_137, primals_139, primals_141, primals_143, primals_144, primals_145, primals_146, primals_148, primals_150, primals_152, primals_153, primals_154, primals_155, primals_157, primals_159, primals_161, primals_162, primals_163, primals_164, primals_166, primals_168, primals_170, primals_171, primals_172, primals_173, primals_175, primals_177, primals_179, primals_180, primals_181, primals_182, primals_184, primals_186, primals_188, primals_189, primals_190, primals_191, primals_193, primals_195, primals_197, primals_198, primals_199, primals_200, primals_202, primals_204, primals_206, primals_207, primals_208, primals_209, primals_211, primals_213, primals_215, primals_216, primals_217, primals_218, primals_220, primals_222, primals_224, primals_225, primals_414, convolution, squeeze_1, mul_7, convolution_1, squeeze_4, clamp_max, convolution_2, squeeze_7, add_14, convolution_3, squeeze_10, mul_29, convolution_4, squeeze_13, clamp_max_1, convolution_5, squeeze_16, add_29, convolution_6, squeeze_19, mul_51, convolution_7, squeeze_22, clamp_max_2, convolution_8, squeeze_25, cat, convolution_9, squeeze_28, mul_73, convolution_10, squeeze_31, add_55, mean, convolution_11, relu, convolution_12, clamp_max_3, convolution_13, squeeze_37, add_65, convolution_14, squeeze_40, mul_103, convolution_15, squeeze_43, add_75, mean_1, convolution_16, relu_1, convolution_17, clamp_max_4, convolution_18, squeeze_49, cat_1, convolution_19, squeeze_52, mul_133, convolution_20, squeeze_55, add_96, mean_2, convolution_21, relu_2, convolution_22, clamp_max_5, convolution_23, squeeze_61, add_106, convolution_24, squeeze_64, mul_163, convolution_25, squeeze_67, add_116, mean_3, convolution_26, relu_3, convolution_27, clamp_max_6, convolution_28, squeeze_73, cat_2, convolution_29, squeeze_76, mul_193, convolution_30, squeeze_79, add_137, mean_4, convolution_31, relu_4, convolution_32, clamp_max_7, convolution_33, squeeze_85, cat_3, convolution_34, squeeze_88, mul_223, convolution_35, squeeze_91, add_158, mean_5, convolution_36, relu_5, convolution_37, clamp_max_8, convolution_38, squeeze_97, cat_4, convolution_39, squeeze_100, mul_253, convolution_40, squeeze_103, add_179, mean_6, convolution_41, relu_6, convolution_42, clamp_max_9, convolution_43, squeeze_109, cat_5, convolution_44, squeeze_112, mul_283, convolution_45, squeeze_115, add_200, mean_7, convolution_46, relu_7, convolution_47, clamp_max_10, convolution_48, squeeze_121, cat_6, convolution_49, squeeze_124, mul_313, convolution_50, squeeze_127, add_221, mean_8, convolution_51, relu_8, convolution_52, clamp_max_11, convolution_53, squeeze_133, add_231, convolution_54, squeeze_136, mul_343, convolution_55, squeeze_139, add_241, mean_9, convolution_56, relu_9, convolution_57, clamp_max_12, convolution_58, squeeze_145, cat_7, convolution_59, squeeze_148, mul_373, convolution_60, squeeze_151, add_262, mean_10, convolution_61, relu_10, convolution_62, clamp_max_13, convolution_63, squeeze_157, cat_8, convolution_64, squeeze_160, mul_403, convolution_65, squeeze_163, add_283, mean_11, convolution_66, relu_11, convolution_67, clamp_max_14, convolution_68, squeeze_169, cat_9, convolution_69, squeeze_172, mul_433, convolution_70, squeeze_175, add_304, mean_12, convolution_71, relu_12, convolution_72, clamp_max_15, convolution_73, squeeze_181, cat_10, convolution_74, squeeze_184, clone_17, permute_1, mul_465, unsqueeze_250, unsqueeze_262, unsqueeze_286, mul_508, unsqueeze_298, unsqueeze_310, unsqueeze_334, mul_551, unsqueeze_346, unsqueeze_358, unsqueeze_382, mul_594, unsqueeze_394, unsqueeze_406, unsqueeze_430, mul_637, unsqueeze_442, unsqueeze_454, unsqueeze_478, mul_680, unsqueeze_490, unsqueeze_502, unsqueeze_526, mul_723, unsqueeze_538, unsqueeze_550, unsqueeze_574, mul_766, unsqueeze_586, unsqueeze_598, unsqueeze_622, mul_809, unsqueeze_634, unsqueeze_646, unsqueeze_670, mul_852, unsqueeze_682, unsqueeze_694, unsqueeze_718, mul_895, unsqueeze_730, unsqueeze_742, unsqueeze_766, mul_938, unsqueeze_778, unsqueeze_790, unsqueeze_814, mul_981, unsqueeze_826, unsqueeze_838, unsqueeze_862, mul_1024, unsqueeze_874, unsqueeze_886, bitwise_or_13, unsqueeze_898, mul_1054, unsqueeze_910, unsqueeze_922, bitwise_or_14, unsqueeze_934, mul_1084, unsqueeze_946, unsqueeze_958, bitwise_or_15, unsqueeze_970, mul_1114, unsqueeze_982, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('rexnet_100', benchmark_compiled_module)
