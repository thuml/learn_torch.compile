
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


cpp_fused__native_batch_norm_legit_constant_pad_nd_convolution_0 = async_compile.cpp('''
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
                       const float* in_ptr50,
                       const float* in_ptr51,
                       const float* in_ptr52,
                       const float* in_ptr53,
                       const float* in_ptr54,
                       const float* in_ptr55,
                       const float* in_ptr56,
                       const float* in_ptr57,
                       const float* in_ptr58,
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
                       float* out_ptr38,
                       float* out_ptr39,
                       float* out_ptr40,
                       float* out_ptr41,
                       float* out_ptr42,
                       float* out_ptr43,
                       float* out_ptr44,
                       float* out_ptr45,
                       float* out_ptr46,
                       float* out_ptr47,
                       float* out_ptr48,
                       float* out_ptr49,
                       float* out_ptr50,
                       float* out_ptr51,
                       float* out_ptr52,
                       float* out_ptr53,
                       float* out_ptr54,
                       float* out_ptr55,
                       float* out_ptr56,
                       float* out_ptr57,
                       float* out_ptr58,
                       float* out_ptr59,
                       float* out_ptr60,
                       float* out_ptr61,
                       float* out_ptr62,
                       float* out_ptr63,
                       float* out_ptr64,
                       float* out_ptr65,
                       float* out_ptr66,
                       float* out_ptr67,
                       float* out_ptr68,
                       float* out_ptr69,
                       float* out_ptr70,
                       float* out_ptr71,
                       float* out_ptr72,
                       float* out_ptr73,
                       float* out_ptr74,
                       float* out_ptr75,
                       float* out_ptr76,
                       float* out_ptr77,
                       float* out_ptr78,
                       float* out_ptr79,
                       float* out_ptr80,
                       float* out_ptr81,
                       float* out_ptr82,
                       float* out_ptr83,
                       float* out_ptr84,
                       float* out_ptr85,
                       float* out_ptr86,
                       float* out_ptr87,
                       float* out_ptr88,
                       float* out_ptr89,
                       float* out_ptr90,
                       float* out_ptr91,
                       float* out_ptr92,
                       float* out_ptr93,
                       float* out_ptr94,
                       float* out_ptr95,
                       float* out_ptr96,
                       float* out_ptr97,
                       float* out_ptr98,
                       float* out_ptr99,
                       float* out_ptr100,
                       float* out_ptr101,
                       float* out_ptr102,
                       float* out_ptr103,
                       float* out_ptr104,
                       float* out_ptr105,
                       float* out_ptr106,
                       float* out_ptr107,
                       float* out_ptr108,
                       float* out_ptr109,
                       float* out_ptr110,
                       float* out_ptr111,
                       float* out_ptr112,
                       float* out_ptr113,
                       float* out_ptr114,
                       float* out_ptr115,
                       float* out_ptr116)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (27L*x0)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                #pragma omp simd simdlen(4)  reduction(welford:tmp_acc0)
                for(long x1=static_cast<long>(24L); x1<static_cast<long>(27L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (27L*x0))];
                    tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                }
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (144L*x0)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (288L*x0)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr5[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (576L*x0)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr6[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr7[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr8[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr9[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (128L*x0)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr10[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr11[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (1152L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr12[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr13[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1 + (1152L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr14[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr15[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1 + (128L*x0)));
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr16[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                        out_ptr17[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                    }
                }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1 + (256L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr18[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr19[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1 + (256L*x0)));
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr20[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                        out_ptr21[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1 + (1152L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr22[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr23[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1 + (1152L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr24[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr25[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x1 + (256L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr26[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr27[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr28[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr29[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x1 + (1152L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr30[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr31[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr16 + static_cast<long>(x1 + (1152L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr32[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr33[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr17 + static_cast<long>(x1 + (256L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr34[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr35[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr18 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr36[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr37[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr19 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr38[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr39[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr20 + static_cast<long>(x1 + (1152L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr40[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr41[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr21 + static_cast<long>(x1 + (1152L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr42[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr43[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr22 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr44[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr45[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr23 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr46[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr47[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr24 + static_cast<long>(x1 + (1152L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr48[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr49[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr25 + static_cast<long>(x1 + (1152L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr50[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr51[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr26 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr52[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr53[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr27 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr54[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr55[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr28 + static_cast<long>(x1 + (1152L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr56[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr57[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr29 + static_cast<long>(x1 + (1152L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr58[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr59[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr30 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr60[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr61[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr31 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr62[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr63[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr32 + static_cast<long>(x1 + (1152L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr64[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr65[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr33 + static_cast<long>(x1 + (1152L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr66[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr67[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr34 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr68[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr69[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr35 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr70[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr71[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr36 + static_cast<long>(x1 + (1152L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr72[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr73[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr37 + static_cast<long>(x1 + (1152L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr74[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr75[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr38 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr76[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr77[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr39 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr78[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr79[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr40 + static_cast<long>(x1 + (1152L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr80[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr81[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr41 + static_cast<long>(x1 + (1152L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr82[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr83[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr42 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr84[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr85[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr43 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr86[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr87[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr44 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr88[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr89[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr45 + static_cast<long>(x1 + (1152L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr90[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr91[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr46 + static_cast<long>(x1 + (1152L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr92[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr93[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr47 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr94[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr95[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr48 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr96[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr97[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr49 + static_cast<long>(x1 + (1152L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr98[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr99[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr50 + static_cast<long>(x1 + (1152L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr100[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr101[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr51 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr102[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr103[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr52 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr104[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr105[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr53 + static_cast<long>(x1 + (1152L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr106[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr107[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr54 + static_cast<long>(x1 + (1152L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr108[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr109[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr55 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr110[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr111[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr56 + static_cast<long>(x1 + (1536L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr112[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr113[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(257L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(257L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x2);
                            auto tmp1 = static_cast<long>(256);
                            auto tmp2 = tmp0 < tmp1;
                            auto tmp3 = c10::convert<long>(x3);
                            auto tmp4 = tmp3 < tmp1;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = in_ptr57[static_cast<long>(x3 + (256L*x2) + (65536L*x1) + (196608L*x0))];
                                return tmp7;
                            }
                            ;
                            auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            out_ptr114[static_cast<long>(x1 + (3L*x3) + (771L*x2) + (198147L*x0))] = tmp8;
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (27L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp4 = out_ptr1[static_cast<long>(x0)];
                        auto tmp12 = in_ptr58[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(27.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = static_cast<float>(0.19245008972987526);
                        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp11 * tmp15;
                        tmp16.store(out_ptr115 + static_cast<long>(x1 + (27L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x1=static_cast<long>(24L); x1<static_cast<long>(27L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (27L*x0))];
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp3 = out_ptr1[static_cast<long>(x0)];
                        auto tmp10 = in_ptr58[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp4 = static_cast<float>(27.0);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = 1 / std::sqrt(tmp7);
                        auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                        auto tmp11 = static_cast<float>(0.19245008972987526);
                        auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                        auto tmp13 = decltype(tmp9)(tmp9 * tmp12);
                        out_ptr115[static_cast<long>(x1 + (27L*x0))] = tmp13;
                    }
                }
            }
        }
        #pragma omp single
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
                            auto tmp0 = out_ptr115[static_cast<long>(x2 + (9L*x1) + (27L*x0))];
                            out_ptr116[static_cast<long>(x1 + (3L*x2) + (27L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_convolution_gelu_mul_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                tmp14.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(144L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (144L*x0)));
                        auto tmp1 = in_ptr1[static_cast<long>(x0)];
                        auto tmp4 = in_ptr2[static_cast<long>(x0)];
                        auto tmp12 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(144.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = static_cast<float>(0.08333333333333333);
                        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp11 * tmp15;
                        tmp16.store(out_ptr0 + static_cast<long>(x1 + (144L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (16L*x2) + (144L*x0)), static_cast<long>(16L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr1 + static_cast<long>(x1 + (16L*x2) + (144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_convolution_gelu_mul_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                tmp14.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (288L*x0)));
                        auto tmp1 = in_ptr1[static_cast<long>(x0)];
                        auto tmp4 = in_ptr2[static_cast<long>(x0)];
                        auto tmp12 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(288.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = static_cast<float>(0.05892556509887896);
                        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp11 * tmp15;
                        tmp16.store(out_ptr0 + static_cast<long>(x1 + (288L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (32L*x2) + (288L*x0)), static_cast<long>(32L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr1 + static_cast<long>(x1 + (32L*x2) + (288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(129L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(129L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(128);
                            auto tmp2 = tmp0 < tmp1;
                            auto tmp3 = c10::convert<int>(x2);
                            auto tmp4 = tmp3 < tmp1;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = masked_load(in_ptr0 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (1048576L*x0)), to_float_mask(tmp5));
                                auto tmp8 = static_cast<float>(0.5);
                                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                                auto tmp10 = tmp7 * tmp9;
                                auto tmp11 = static_cast<float>(0.7071067811865476);
                                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                                auto tmp13 = tmp7 * tmp12;
                                auto tmp14 = tmp13.erf();
                                auto tmp15 = static_cast<float>(1.0);
                                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                                auto tmp17 = tmp14 + tmp16;
                                auto tmp18 = tmp10 * tmp17;
                                auto tmp19 = static_cast<float>(1.7015043497085571);
                                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                                auto tmp21 = tmp18 * tmp20;
                                return tmp21;
                            }
                            ;
                            auto tmp22 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                            tmp22.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (8256L*x1) + (1065024L*x0)));
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(576L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (576L*x0)));
                        auto tmp1 = in_ptr2[static_cast<long>(x0)];
                        auto tmp4 = in_ptr3[static_cast<long>(x0)];
                        auto tmp12 = in_ptr4[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(576.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = static_cast<float>(0.041666666666666664);
                        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp11 * tmp15;
                        tmp16.store(out_ptr1 + static_cast<long>(x1 + (576L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr2 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_gelu_mul_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp14 * tmp9;
                tmp15.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = in_ptr1[static_cast<long>(x0)];
                        auto tmp4 = in_ptr2[static_cast<long>(x0)];
                        auto tmp12 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(128.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = static_cast<float>(0.08838834764831845);
                        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp11 * tmp15;
                        tmp16.store(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_convolution_gelu_mul_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                tmp14.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1152.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.02946278254943948);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_convolution_gelu_mul_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                tmp14.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1152.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.02946278254943948);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_gelu_mul_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                tmp14.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = in_ptr1[static_cast<long>(x0)];
                        auto tmp4 = in_ptr2[static_cast<long>(x0)];
                        auto tmp12 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(128.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = static_cast<float>(0.08838834764831845);
                        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp11 * tmp15;
                        tmp16.store(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_mean_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(4096L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x2) + (1048576L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(4096.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = in_ptr1[static_cast<long>(x0)];
                auto tmp4 = in_ptr2[static_cast<long>(x0)];
                auto tmp12 = in_ptr3[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 - tmp2;
                auto tmp5 = static_cast<float>(128.0);
                auto tmp6 = tmp4 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                auto tmp9 = 1 / std::sqrt(tmp8);
                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                auto tmp11 = tmp3 * tmp10;
                auto tmp13 = static_cast<float>(0.08838834764831845);
                auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp11 * tmp15;
                tmp16.store(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_add_gelu_mul_sigmoid_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (1048576L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x0)));
                        auto tmp7 = in_ptr1[static_cast<long>(0L)];
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (256L*x1) + (1048576L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp10 = static_cast<float>(0.2);
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp14 = tmp12 + tmp13;
                        auto tmp15 = static_cast<float>(0.5);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp14 * tmp16;
                        auto tmp18 = static_cast<float>(0.7071067811865476);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp14 * tmp19;
                        auto tmp21 = tmp20.erf();
                        auto tmp22 = static_cast<float>(1.0);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 + tmp23;
                        auto tmp25 = tmp17 * tmp24;
                        auto tmp26 = static_cast<float>(1.7015043497085571);
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp25 * tmp27;
                        auto tmp29 = static_cast<float>(0.9805806756909201);
                        auto tmp30 = at::vec::Vectorized<float>(tmp29);
                        auto tmp31 = tmp28 * tmp30;
                        tmp31.store(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (1048576L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = in_ptr4[static_cast<long>(x0)];
                        auto tmp4 = in_ptr5[static_cast<long>(x0)];
                        auto tmp12 = in_ptr6[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(256.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = static_cast<float>(0.0625);
                        auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp11 * tmp15;
                        tmp16.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(65L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(65L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(64);
                            auto tmp2 = tmp0 < tmp1;
                            auto tmp3 = c10::convert<int>(x2);
                            auto tmp4 = tmp3 < tmp1;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = masked_load(in_ptr0 + static_cast<long>(x3 + (256L*x2) + (16384L*x1) + (1048576L*x0)), to_float_mask(tmp5));
                                auto tmp8 = static_cast<float>(0.5);
                                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                                auto tmp10 = tmp7 * tmp9;
                                auto tmp11 = static_cast<float>(0.7071067811865476);
                                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                                auto tmp13 = tmp7 * tmp12;
                                auto tmp14 = tmp13.erf();
                                auto tmp15 = static_cast<float>(1.0);
                                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                                auto tmp17 = tmp14 + tmp16;
                                auto tmp18 = tmp10 * tmp17;
                                auto tmp19 = static_cast<float>(1.7015043497085571);
                                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                                auto tmp21 = tmp18 * tmp20;
                                return tmp21;
                            }
                            ;
                            auto tmp22 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                            tmp22.store(out_ptr0 + static_cast<long>(x3 + (256L*x2) + (16640L*x1) + (1081600L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = in_ptr2[static_cast<long>(x0)];
                    auto tmp4 = in_ptr3[static_cast<long>(x0)];
                    auto tmp12 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1152.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.02946278254943948);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr1 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr2 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_convolution_gelu_mul_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                tmp14.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1152.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.02946278254943948);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_gelu_mul_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                tmp14.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(256.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.0625);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mean_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x2) + (524288L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
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


cpp_fused_relu_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_avg_pool2d_17 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (32768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(256L + x2 + (512L*x1) + (32768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(16384L + x2 + (512L*x1) + (32768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(16640L + x2 + (512L*x1) + (32768L*x0)));
                        auto tmp2 = tmp1 + tmp0;
                        auto tmp4 = tmp3 + tmp2;
                        auto tmp6 = tmp5 + tmp4;
                        auto tmp7 = static_cast<float>(0.25);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        tmp9.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (8192L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr2[static_cast<long>(x0)];
                    auto tmp4 = in_ptr3[static_cast<long>(x0)];
                    auto tmp12 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(256.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.0625);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_add_gelu_mul_sigmoid_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (524288L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp7 = in_ptr2[static_cast<long>(0L)];
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*x1) + (524288L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp10 = static_cast<float>(0.2);
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp14 = tmp12 + tmp13;
                        auto tmp15 = static_cast<float>(0.5);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp14 * tmp16;
                        auto tmp18 = static_cast<float>(0.7071067811865476);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp14 * tmp19;
                        auto tmp21 = tmp20.erf();
                        auto tmp22 = static_cast<float>(1.0);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 + tmp23;
                        auto tmp25 = tmp17 * tmp24;
                        auto tmp26 = static_cast<float>(1.7015043497085571);
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp25 * tmp27;
                        auto tmp29 = static_cast<float>(0.9805806756909201);
                        auto tmp30 = at::vec::Vectorized<float>(tmp29);
                        auto tmp31 = tmp28 * tmp30;
                        tmp31.store(out_ptr0 + static_cast<long>(x2 + (512L*x1) + (524288L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp4 = in_ptr6[static_cast<long>(x0)];
                    auto tmp12 = in_ptr7[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(512.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.04419417382415922);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_convolution_gelu_mul_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                tmp14.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1152.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.02946278254943948);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_convolution_gelu_mul_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                tmp14.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1152.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.02946278254943948);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_gelu_mul_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                tmp14.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(256.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.0625);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mean_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x2) + (524288L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
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


cpp_fused_relu_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_add_gelu_mul_sigmoid_24 = async_compile.cpp('''
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
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (524288L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp7 = in_ptr2[static_cast<long>(0L)];
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (524288L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp18 = in_ptr4[static_cast<long>(0L)];
                        auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (512L*x1) + (524288L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp10 = static_cast<float>(0.2);
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp15 = decltype(tmp14)(1)/(decltype(tmp14)(1) + tmp14.neg().exp());
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp17 = tmp16 * tmp5;
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 * tmp19;
                        auto tmp21 = tmp20 * tmp11;
                        auto tmp23 = tmp21 + tmp22;
                        auto tmp24 = tmp12 + tmp23;
                        tmp24.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (524288L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.9622504486493761);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                tmp17.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr7[static_cast<long>(x0)];
                    auto tmp4 = in_ptr8[static_cast<long>(x0)];
                    auto tmp12 = in_ptr9[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(512.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.04419417382415922);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(33L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(33L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(768L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(32);
                            auto tmp2 = tmp0 < tmp1;
                            auto tmp3 = c10::convert<int>(x2);
                            auto tmp4 = tmp3 < tmp1;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = masked_load(in_ptr0 + static_cast<long>(x3 + (768L*x2) + (24576L*x1) + (786432L*x0)), to_float_mask(tmp5));
                                auto tmp8 = static_cast<float>(0.5);
                                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                                auto tmp10 = tmp7 * tmp9;
                                auto tmp11 = static_cast<float>(0.7071067811865476);
                                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                                auto tmp13 = tmp7 * tmp12;
                                auto tmp14 = tmp13.erf();
                                auto tmp15 = static_cast<float>(1.0);
                                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                                auto tmp17 = tmp14 + tmp16;
                                auto tmp18 = tmp10 * tmp17;
                                auto tmp19 = static_cast<float>(1.7015043497085571);
                                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                                auto tmp21 = tmp18 * tmp20;
                                return tmp21;
                            }
                            ;
                            auto tmp22 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                            tmp22.store(out_ptr0 + static_cast<long>(x3 + (768L*x2) + (25344L*x1) + (836352L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = in_ptr2[static_cast<long>(x0)];
                    auto tmp4 = in_ptr3[static_cast<long>(x0)];
                    auto tmp12 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1152.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.02946278254943948);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr1 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr2 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_convolution_gelu_mul_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                tmp14.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1152.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.02946278254943948);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_gelu_mul_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                tmp14.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.03608439182435161);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mean_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x2) + (393216L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_avg_pool2d_30 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1024L*x1) + (32768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x2 + (1024L*x1) + (32768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(16384L + x2 + (1024L*x1) + (32768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(16896L + x2 + (1024L*x1) + (32768L*x0)));
                        auto tmp2 = tmp1 + tmp0;
                        auto tmp4 = tmp3 + tmp2;
                        auto tmp6 = tmp5 + tmp4;
                        auto tmp7 = static_cast<float>(0.25);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        tmp9.store(out_ptr0 + static_cast<long>(x2 + (512L*x1) + (8192L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = in_ptr2[static_cast<long>(x0)];
                    auto tmp4 = in_ptr3[static_cast<long>(x0)];
                    auto tmp12 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(512.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.04419417382415922);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_add_gelu_mul_sigmoid_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (393216L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp7 = in_ptr2[static_cast<long>(0L)];
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1536L*x1) + (393216L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp10 = static_cast<float>(0.2);
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp14 = tmp12 + tmp13;
                        auto tmp15 = static_cast<float>(0.5);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp14 * tmp16;
                        auto tmp18 = static_cast<float>(0.7071067811865476);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp14 * tmp19;
                        auto tmp21 = tmp20.erf();
                        auto tmp22 = static_cast<float>(1.0);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 + tmp23;
                        auto tmp25 = tmp17 * tmp24;
                        auto tmp26 = static_cast<float>(1.7015043497085571);
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp25 * tmp27;
                        auto tmp29 = static_cast<float>(0.9805806756909201);
                        auto tmp30 = at::vec::Vectorized<float>(tmp29);
                        auto tmp31 = tmp28 * tmp30;
                        tmp31.store(out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (393216L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp4 = in_ptr6[static_cast<long>(x0)];
                    auto tmp12 = in_ptr7[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.02551551815399144);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_convolution_gelu_mul_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                tmp14.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1152.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.02946278254943948);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_convolution_gelu_mul_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                tmp14.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1152.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.02946278254943948);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_gelu_mul_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                tmp14.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.03608439182435161);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mean_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x2) + (393216L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_add_gelu_mul_sigmoid_37 = async_compile.cpp('''
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
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (393216L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp7 = in_ptr2[static_cast<long>(0L)];
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (393216L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp18 = in_ptr4[static_cast<long>(0L)];
                        auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (1536L*x1) + (393216L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp10 = static_cast<float>(0.2);
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp15 = decltype(tmp14)(1)/(decltype(tmp14)(1) + tmp14.neg().exp());
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp17 = tmp16 * tmp5;
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 * tmp19;
                        auto tmp21 = tmp20 * tmp11;
                        auto tmp23 = tmp21 + tmp22;
                        auto tmp24 = tmp12 + tmp23;
                        tmp24.store(in_out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (393216L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.9622504486493761);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                tmp17.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = in_ptr7[static_cast<long>(x0)];
                    auto tmp4 = in_ptr8[static_cast<long>(x0)];
                    auto tmp12 = in_ptr9[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.02551551815399144);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_convolution_gelu_mul_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                tmp14.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1152.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.02946278254943948);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_convolution_gelu_mul_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                tmp14.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1152.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.02946278254943948);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_gelu_mul_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                tmp14.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.03608439182435161);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mean_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x2) + (393216L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_add_gelu_mul_sigmoid_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (393216L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp7 = in_ptr2[static_cast<long>(0L)];
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1536L*x1) + (393216L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp10 = static_cast<float>(0.2);
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp14 = tmp12 + tmp13;
                        auto tmp15 = static_cast<float>(0.5);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp14 * tmp16;
                        auto tmp18 = static_cast<float>(0.7071067811865476);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp14 * tmp19;
                        auto tmp21 = tmp20.erf();
                        auto tmp22 = static_cast<float>(1.0);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 + tmp23;
                        auto tmp25 = tmp17 * tmp24;
                        auto tmp26 = static_cast<float>(1.7015043497085571);
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp25 * tmp27;
                        auto tmp29 = static_cast<float>(0.9449111825230679);
                        auto tmp30 = at::vec::Vectorized<float>(tmp29);
                        auto tmp31 = tmp28 * tmp30;
                        tmp31.store(out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (393216L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp4 = in_ptr6[static_cast<long>(x0)];
                    auto tmp12 = in_ptr7[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.02551551815399144);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_convolution_gelu_mul_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                tmp14.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1152.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.02946278254943948);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_convolution_gelu_mul_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                tmp14.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1152.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.02946278254943948);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_gelu_mul_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                tmp14.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.03608439182435161);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mean_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x2) + (393216L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_add_gelu_mul_sigmoid_49 = async_compile.cpp('''
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
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (393216L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp7 = in_ptr2[static_cast<long>(0L)];
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1536L*x1) + (393216L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp18 = in_ptr5[static_cast<long>(0L)];
                        auto tmp22 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (393216L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp10 = static_cast<float>(0.2);
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp15 = decltype(tmp14)(1)/(decltype(tmp14)(1) + tmp14.neg().exp());
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp17 = tmp16 * tmp5;
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 * tmp19;
                        auto tmp21 = tmp20 * tmp11;
                        auto tmp23 = tmp21 + tmp22;
                        auto tmp24 = tmp12 + tmp23;
                        tmp24.store(in_out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (393216L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.9284766908852592);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                tmp17.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = in_ptr7[static_cast<long>(x0)];
                    auto tmp4 = in_ptr8[static_cast<long>(x0)];
                    auto tmp12 = in_ptr9[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.02551551815399144);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_convolution_gelu_mul_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                tmp14.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1152.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.02946278254943948);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_convolution_gelu_mul_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                tmp14.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1152.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.02946278254943948);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_gelu_mul_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                tmp14.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.03608439182435161);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mean_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x2) + (393216L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_add_gelu_mul_sigmoid_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (393216L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp7 = in_ptr2[static_cast<long>(0L)];
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1536L*x1) + (393216L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp10 = static_cast<float>(0.2);
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp14 = tmp12 + tmp13;
                        auto tmp15 = static_cast<float>(0.5);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp14 * tmp16;
                        auto tmp18 = static_cast<float>(0.7071067811865476);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp14 * tmp19;
                        auto tmp21 = tmp20.erf();
                        auto tmp22 = static_cast<float>(1.0);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 + tmp23;
                        auto tmp25 = tmp17 * tmp24;
                        auto tmp26 = static_cast<float>(1.7015043497085571);
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp25 * tmp27;
                        auto tmp29 = static_cast<float>(0.9128709291752768);
                        auto tmp30 = at::vec::Vectorized<float>(tmp29);
                        auto tmp31 = tmp28 * tmp30;
                        tmp31.store(out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (393216L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp4 = in_ptr6[static_cast<long>(x0)];
                    auto tmp12 = in_ptr7[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.02551551815399144);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_convolution_gelu_mul_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                tmp14.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1152.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.02946278254943948);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_convolution_gelu_mul_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                tmp14.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1152.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.02946278254943948);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_gelu_mul_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                tmp14.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.03608439182435161);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mean_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x2) + (393216L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_add_gelu_mul_sigmoid_61 = async_compile.cpp('''
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
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (393216L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp7 = in_ptr2[static_cast<long>(0L)];
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1536L*x1) + (393216L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp18 = in_ptr5[static_cast<long>(0L)];
                        auto tmp22 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (393216L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp10 = static_cast<float>(0.2);
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp15 = decltype(tmp14)(1)/(decltype(tmp14)(1) + tmp14.neg().exp());
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp17 = tmp16 * tmp5;
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 * tmp19;
                        auto tmp21 = tmp20 * tmp11;
                        auto tmp23 = tmp21 + tmp22;
                        auto tmp24 = tmp12 + tmp23;
                        tmp24.store(in_out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (393216L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3145728L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.8980265101338745);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                tmp17.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = in_ptr7[static_cast<long>(x0)];
                    auto tmp4 = in_ptr8[static_cast<long>(x0)];
                    auto tmp12 = in_ptr9[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.02551551815399144);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(17L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(17L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(768L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(16);
                            auto tmp2 = tmp0 < tmp1;
                            auto tmp3 = c10::convert<int>(x2);
                            auto tmp4 = tmp3 < tmp1;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = masked_load(in_ptr0 + static_cast<long>(x3 + (768L*x2) + (12288L*x1) + (196608L*x0)), to_float_mask(tmp5));
                                auto tmp8 = static_cast<float>(0.5);
                                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                                auto tmp10 = tmp7 * tmp9;
                                auto tmp11 = static_cast<float>(0.7071067811865476);
                                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                                auto tmp13 = tmp7 * tmp12;
                                auto tmp14 = tmp13.erf();
                                auto tmp15 = static_cast<float>(1.0);
                                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                                auto tmp17 = tmp14 + tmp16;
                                auto tmp18 = tmp10 * tmp17;
                                auto tmp19 = static_cast<float>(1.7015043497085571);
                                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                                auto tmp21 = tmp18 * tmp20;
                                return tmp21;
                            }
                            ;
                            auto tmp22 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                            tmp22.store(out_ptr0 + static_cast<long>(x3 + (768L*x2) + (13056L*x1) + (221952L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = in_ptr2[static_cast<long>(x0)];
                    auto tmp4 = in_ptr3[static_cast<long>(x0)];
                    auto tmp12 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1152.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.02946278254943948);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr1 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr2 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_convolution_gelu_mul_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                tmp14.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1152.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.02946278254943948);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_gelu_mul_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                tmp14.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.03608439182435161);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mean_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x2) + (98304L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(64.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_avg_pool2d_67 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (3072L*x1) + (49152L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(1536L + x2 + (3072L*x1) + (49152L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(24576L + x2 + (3072L*x1) + (49152L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(26112L + x2 + (3072L*x1) + (49152L*x0)));
                        auto tmp2 = tmp1 + tmp0;
                        auto tmp4 = tmp3 + tmp2;
                        auto tmp6 = tmp5 + tmp4;
                        auto tmp7 = static_cast<float>(0.25);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        tmp9.store(out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (12288L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = in_ptr2[static_cast<long>(x0)];
                    auto tmp4 = in_ptr3[static_cast<long>(x0)];
                    auto tmp12 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.02551551815399144);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_add_gelu_mul_sigmoid_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (98304L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp7 = in_ptr2[static_cast<long>(0L)];
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1536L*x1) + (98304L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp10 = static_cast<float>(0.2);
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp14 = tmp12 + tmp13;
                        auto tmp15 = static_cast<float>(0.5);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp14 * tmp16;
                        auto tmp18 = static_cast<float>(0.7071067811865476);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp14 * tmp19;
                        auto tmp21 = tmp20.erf();
                        auto tmp22 = static_cast<float>(1.0);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 + tmp23;
                        auto tmp25 = tmp17 * tmp24;
                        auto tmp26 = static_cast<float>(1.7015043497085571);
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp25 * tmp27;
                        auto tmp29 = static_cast<float>(0.9805806756909201);
                        auto tmp30 = at::vec::Vectorized<float>(tmp29);
                        auto tmp31 = tmp28 * tmp30;
                        tmp31.store(out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (98304L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp4 = in_ptr6[static_cast<long>(x0)];
                    auto tmp12 = in_ptr7[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.02551551815399144);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_convolution_gelu_mul_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                tmp14.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1152.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.02946278254943948);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_convolution_gelu_mul_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                tmp14.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1152.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.02946278254943948);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_gelu_mul_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                tmp14.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.03608439182435161);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mean_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x2) + (98304L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(64.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_add_gelu_mul_sigmoid_74 = async_compile.cpp('''
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
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (98304L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp7 = in_ptr2[static_cast<long>(0L)];
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (98304L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp18 = in_ptr4[static_cast<long>(0L)];
                        auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (1536L*x1) + (98304L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp10 = static_cast<float>(0.2);
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp15 = decltype(tmp14)(1)/(decltype(tmp14)(1) + tmp14.neg().exp());
                        auto tmp16 = tmp13 * tmp15;
                        auto tmp17 = tmp16 * tmp5;
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 * tmp19;
                        auto tmp21 = tmp20 * tmp11;
                        auto tmp23 = tmp21 + tmp22;
                        auto tmp24 = tmp12 + tmp23;
                        tmp24.store(in_out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (98304L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(786432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = static_cast<float>(0.9622504486493761);
                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                auto tmp17 = tmp14 * tmp16;
                tmp17.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = in_ptr7[static_cast<long>(x0)];
                    auto tmp4 = in_ptr8[static_cast<long>(x0)];
                    auto tmp12 = in_ptr9[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.02551551815399144);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr1 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_convolution_gelu_mul_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                tmp14.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1152.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.02946278254943948);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_convolution_gelu_mul_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                tmp14.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1152L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1152.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.02946278254943948);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (1152L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_gelu_mul_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                auto tmp12 = static_cast<float>(1.7015043497085571);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                tmp14.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_ptr2[static_cast<long>(x0)];
                    auto tmp12 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.03608439182435161);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_mean_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x2) + (98304L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(64.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_add_mul_sigmoid_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1536L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1536L*x1) + (98304L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1536L*x0)));
                        auto tmp7 = in_ptr2[static_cast<long>(0L)];
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (98304L*x0)));
                        auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(2.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 * tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        auto tmp10 = static_cast<float>(0.2);
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp14 = tmp12 + tmp13;
                        tmp14.store(in_out_ptr0 + static_cast<long>(x2 + (1536L*x1) + (98304L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1536L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp4 = in_ptr5[static_cast<long>(x0)];
                    auto tmp12 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1536.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = static_cast<float>(0.02551551815399144);
                    auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 * tmp15;
                    tmp16.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_mean_mul_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3072L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (3072L*x2) + (196608L*x0)));
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
                            auto tmp12 = static_cast<float>(1.7015043497085571);
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 * tmp13;
                            tmp_acc0_vec = tmp_acc0_vec + tmp14;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (3072L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(24576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(64.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (16, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg2_1, (16, ), (1, ))
    assert_size_stride(arg3_1, (32, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg4_1, (32, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg5_1, (32, ), (1, ))
    assert_size_stride(arg6_1, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg7_1, (64, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg8_1, (64, ), (1, ))
    assert_size_stride(arg9_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg10_1, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg11_1, (128, ), (1, ))
    assert_size_stride(arg12_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg13_1, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg14_1, (256, ), (1, ))
    assert_size_stride(arg15_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg16_1, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg17_1, (128, ), (1, ))
    assert_size_stride(arg18_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg19_1, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg20_1, (128, ), (1, ))
    assert_size_stride(arg21_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg22_1, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg23_1, (128, ), (1, ))
    assert_size_stride(arg24_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg25_1, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg26_1, (256, ), (1, ))
    assert_size_stride(arg27_1, (), ())
    assert_size_stride(arg28_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg29_1, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg30_1, (512, ), (1, ))
    assert_size_stride(arg31_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg32_1, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg33_1, (256, ), (1, ))
    assert_size_stride(arg34_1, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg35_1, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg36_1, (256, ), (1, ))
    assert_size_stride(arg37_1, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg38_1, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg39_1, (256, ), (1, ))
    assert_size_stride(arg40_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg41_1, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg42_1, (512, ), (1, ))
    assert_size_stride(arg43_1, (), ())
    assert_size_stride(arg44_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg45_1, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg46_1, (256, ), (1, ))
    assert_size_stride(arg47_1, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg48_1, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg49_1, (256, ), (1, ))
    assert_size_stride(arg50_1, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg51_1, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg52_1, (256, ), (1, ))
    assert_size_stride(arg53_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg54_1, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg55_1, (512, ), (1, ))
    assert_size_stride(arg56_1, (), ())
    assert_size_stride(arg57_1, (1536, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg58_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg59_1, (1536, ), (1, ))
    assert_size_stride(arg60_1, (768, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg61_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg62_1, (768, ), (1, ))
    assert_size_stride(arg63_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg64_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg65_1, (768, ), (1, ))
    assert_size_stride(arg66_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg67_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg68_1, (768, ), (1, ))
    assert_size_stride(arg69_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg70_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg71_1, (1536, ), (1, ))
    assert_size_stride(arg72_1, (), ())
    assert_size_stride(arg73_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg74_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg75_1, (768, ), (1, ))
    assert_size_stride(arg76_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg77_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg78_1, (768, ), (1, ))
    assert_size_stride(arg79_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg80_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg81_1, (768, ), (1, ))
    assert_size_stride(arg82_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg83_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg84_1, (1536, ), (1, ))
    assert_size_stride(arg85_1, (), ())
    assert_size_stride(arg86_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg87_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg88_1, (768, ), (1, ))
    assert_size_stride(arg89_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg90_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg91_1, (768, ), (1, ))
    assert_size_stride(arg92_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg93_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg94_1, (768, ), (1, ))
    assert_size_stride(arg95_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg96_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg97_1, (1536, ), (1, ))
    assert_size_stride(arg98_1, (), ())
    assert_size_stride(arg99_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg100_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg101_1, (768, ), (1, ))
    assert_size_stride(arg102_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg103_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg104_1, (768, ), (1, ))
    assert_size_stride(arg105_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg106_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg107_1, (768, ), (1, ))
    assert_size_stride(arg108_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg109_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg110_1, (1536, ), (1, ))
    assert_size_stride(arg111_1, (), ())
    assert_size_stride(arg112_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg113_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg114_1, (768, ), (1, ))
    assert_size_stride(arg115_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg116_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg117_1, (768, ), (1, ))
    assert_size_stride(arg118_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg119_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg120_1, (768, ), (1, ))
    assert_size_stride(arg121_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg122_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg123_1, (1536, ), (1, ))
    assert_size_stride(arg124_1, (), ())
    assert_size_stride(arg125_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg126_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg127_1, (768, ), (1, ))
    assert_size_stride(arg128_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg129_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg130_1, (768, ), (1, ))
    assert_size_stride(arg131_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg132_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg133_1, (768, ), (1, ))
    assert_size_stride(arg134_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg135_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg136_1, (1536, ), (1, ))
    assert_size_stride(arg137_1, (), ())
    assert_size_stride(arg138_1, (1536, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg139_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg140_1, (1536, ), (1, ))
    assert_size_stride(arg141_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg142_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg143_1, (768, ), (1, ))
    assert_size_stride(arg144_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg145_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg146_1, (768, ), (1, ))
    assert_size_stride(arg147_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg148_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg149_1, (768, ), (1, ))
    assert_size_stride(arg150_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg151_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg152_1, (1536, ), (1, ))
    assert_size_stride(arg153_1, (), ())
    assert_size_stride(arg154_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg155_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg156_1, (768, ), (1, ))
    assert_size_stride(arg157_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg158_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg159_1, (768, ), (1, ))
    assert_size_stride(arg160_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg161_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg162_1, (768, ), (1, ))
    assert_size_stride(arg163_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg164_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg165_1, (1536, ), (1, ))
    assert_size_stride(arg166_1, (), ())
    assert_size_stride(arg167_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg168_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg169_1, (768, ), (1, ))
    assert_size_stride(arg170_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg171_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg172_1, (768, ), (1, ))
    assert_size_stride(arg173_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg174_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg175_1, (768, ), (1, ))
    assert_size_stride(arg176_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg177_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg178_1, (1536, ), (1, ))
    assert_size_stride(arg179_1, (), ())
    assert_size_stride(arg180_1, (3072, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg181_1, (3072, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg182_1, (3072, ), (1, ))
    assert_size_stride(arg183_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg184_1, (128, ), (1, ))
    assert_size_stride(arg185_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg186_1, (256, ), (1, ))
    assert_size_stride(arg187_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg188_1, (256, ), (1, ))
    assert_size_stride(arg189_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg190_1, (512, ), (1, ))
    assert_size_stride(arg191_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg192_1, (256, ), (1, ))
    assert_size_stride(arg193_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg194_1, (512, ), (1, ))
    assert_size_stride(arg195_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg196_1, (768, ), (1, ))
    assert_size_stride(arg197_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg198_1, (1536, ), (1, ))
    assert_size_stride(arg199_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg200_1, (768, ), (1, ))
    assert_size_stride(arg201_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg202_1, (1536, ), (1, ))
    assert_size_stride(arg203_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg204_1, (768, ), (1, ))
    assert_size_stride(arg205_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg206_1, (1536, ), (1, ))
    assert_size_stride(arg207_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg208_1, (768, ), (1, ))
    assert_size_stride(arg209_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg210_1, (1536, ), (1, ))
    assert_size_stride(arg211_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg212_1, (768, ), (1, ))
    assert_size_stride(arg213_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg214_1, (1536, ), (1, ))
    assert_size_stride(arg215_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg216_1, (768, ), (1, ))
    assert_size_stride(arg217_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg218_1, (1536, ), (1, ))
    assert_size_stride(arg219_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg220_1, (768, ), (1, ))
    assert_size_stride(arg221_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg222_1, (1536, ), (1, ))
    assert_size_stride(arg223_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg224_1, (768, ), (1, ))
    assert_size_stride(arg225_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg226_1, (1536, ), (1, ))
    assert_size_stride(arg227_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg228_1, (768, ), (1, ))
    assert_size_stride(arg229_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg230_1, (1536, ), (1, ))
    assert_size_stride(arg231_1, (1000, 3072), (3072, 1))
    assert_size_stride(arg232_1, (1000, ), (1, ))
    assert_size_stride(arg233_1, (8, 3, 256, 256), (196608, 65536, 256, 1))
    buf0 = empty_strided((1, 16, 1), (16, 1, 16), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((1, 16, 1), (16, 1, 16), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((1, 32, 1), (32, 1, 32), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((1, 32, 1), (32, 1, 32), device='cpu', dtype=torch.float32)
    buf6 = empty_strided((1, 64, 1), (64, 1, 64), device='cpu', dtype=torch.float32)
    buf7 = empty_strided((1, 64, 1), (64, 1, 64), device='cpu', dtype=torch.float32)
    buf9 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf10 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf12 = empty_strided((1, 256, 1), (256, 1, 256), device='cpu', dtype=torch.float32)
    buf13 = empty_strided((1, 256, 1), (256, 1, 256), device='cpu', dtype=torch.float32)
    buf15 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf16 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf18 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf19 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf21 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf22 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf24 = empty_strided((1, 256, 1), (256, 1, 256), device='cpu', dtype=torch.float32)
    buf25 = empty_strided((1, 256, 1), (256, 1, 256), device='cpu', dtype=torch.float32)
    buf27 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf28 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf30 = empty_strided((1, 256, 1), (256, 1, 256), device='cpu', dtype=torch.float32)
    buf31 = empty_strided((1, 256, 1), (256, 1, 256), device='cpu', dtype=torch.float32)
    buf33 = empty_strided((1, 256, 1), (256, 1, 256), device='cpu', dtype=torch.float32)
    buf34 = empty_strided((1, 256, 1), (256, 1, 256), device='cpu', dtype=torch.float32)
    buf36 = empty_strided((1, 256, 1), (256, 1, 256), device='cpu', dtype=torch.float32)
    buf37 = empty_strided((1, 256, 1), (256, 1, 256), device='cpu', dtype=torch.float32)
    buf39 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf40 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf42 = empty_strided((1, 256, 1), (256, 1, 256), device='cpu', dtype=torch.float32)
    buf43 = empty_strided((1, 256, 1), (256, 1, 256), device='cpu', dtype=torch.float32)
    buf45 = empty_strided((1, 256, 1), (256, 1, 256), device='cpu', dtype=torch.float32)
    buf46 = empty_strided((1, 256, 1), (256, 1, 256), device='cpu', dtype=torch.float32)
    buf48 = empty_strided((1, 256, 1), (256, 1, 256), device='cpu', dtype=torch.float32)
    buf49 = empty_strided((1, 256, 1), (256, 1, 256), device='cpu', dtype=torch.float32)
    buf51 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf52 = empty_strided((1, 512, 1), (512, 1, 512), device='cpu', dtype=torch.float32)
    buf54 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cpu', dtype=torch.float32)
    buf55 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cpu', dtype=torch.float32)
    buf57 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf58 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf60 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf61 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf63 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf64 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf66 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cpu', dtype=torch.float32)
    buf67 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cpu', dtype=torch.float32)
    buf69 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf70 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf72 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf73 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf75 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf76 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf78 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cpu', dtype=torch.float32)
    buf79 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cpu', dtype=torch.float32)
    buf81 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf82 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf84 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf85 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf87 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf88 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf90 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cpu', dtype=torch.float32)
    buf91 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cpu', dtype=torch.float32)
    buf93 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf94 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf96 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf97 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf99 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf100 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf102 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cpu', dtype=torch.float32)
    buf103 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cpu', dtype=torch.float32)
    buf105 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf106 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf108 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf109 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf111 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf112 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf114 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cpu', dtype=torch.float32)
    buf115 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cpu', dtype=torch.float32)
    buf117 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf118 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf120 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf121 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf123 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf124 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf126 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cpu', dtype=torch.float32)
    buf127 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cpu', dtype=torch.float32)
    buf129 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cpu', dtype=torch.float32)
    buf130 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cpu', dtype=torch.float32)
    buf132 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf133 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf135 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf136 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf138 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf139 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf141 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cpu', dtype=torch.float32)
    buf142 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cpu', dtype=torch.float32)
    buf144 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf145 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf147 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf148 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf150 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf151 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf153 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cpu', dtype=torch.float32)
    buf154 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cpu', dtype=torch.float32)
    buf156 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf157 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf159 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf160 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf162 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf163 = empty_strided((1, 768, 1), (768, 1, 768), device='cpu', dtype=torch.float32)
    buf165 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cpu', dtype=torch.float32)
    buf166 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cpu', dtype=torch.float32)
    buf168 = empty_strided((1, 3072, 1), (3072, 1, 3072), device='cpu', dtype=torch.float32)
    buf169 = empty_strided((1, 3072, 1), (3072, 1, 3072), device='cpu', dtype=torch.float32)
    buf171 = empty_strided((8, 3, 257, 257), (198147, 1, 771, 3), device='cpu', dtype=torch.float32)
    buf172 = empty((1, 16, 27), device='cpu', dtype=torch.float32)
    buf173 = empty_strided((16, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_constant_pad_nd_convolution_0(c_void_p(arg0_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(arg9_1.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(arg24_1.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(arg31_1.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(arg37_1.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(arg44_1.data_ptr()), c_void_p(arg47_1.data_ptr()), c_void_p(arg50_1.data_ptr()), c_void_p(arg53_1.data_ptr()), c_void_p(arg57_1.data_ptr()), c_void_p(arg60_1.data_ptr()), c_void_p(arg63_1.data_ptr()), c_void_p(arg66_1.data_ptr()), c_void_p(arg69_1.data_ptr()), c_void_p(arg73_1.data_ptr()), c_void_p(arg76_1.data_ptr()), c_void_p(arg79_1.data_ptr()), c_void_p(arg82_1.data_ptr()), c_void_p(arg86_1.data_ptr()), c_void_p(arg89_1.data_ptr()), c_void_p(arg92_1.data_ptr()), c_void_p(arg95_1.data_ptr()), c_void_p(arg99_1.data_ptr()), c_void_p(arg102_1.data_ptr()), c_void_p(arg105_1.data_ptr()), c_void_p(arg108_1.data_ptr()), c_void_p(arg112_1.data_ptr()), c_void_p(arg115_1.data_ptr()), c_void_p(arg118_1.data_ptr()), c_void_p(arg121_1.data_ptr()), c_void_p(arg125_1.data_ptr()), c_void_p(arg128_1.data_ptr()), c_void_p(arg131_1.data_ptr()), c_void_p(arg134_1.data_ptr()), c_void_p(arg138_1.data_ptr()), c_void_p(arg141_1.data_ptr()), c_void_p(arg144_1.data_ptr()), c_void_p(arg147_1.data_ptr()), c_void_p(arg150_1.data_ptr()), c_void_p(arg154_1.data_ptr()), c_void_p(arg157_1.data_ptr()), c_void_p(arg160_1.data_ptr()), c_void_p(arg163_1.data_ptr()), c_void_p(arg167_1.data_ptr()), c_void_p(arg170_1.data_ptr()), c_void_p(arg173_1.data_ptr()), c_void_p(arg176_1.data_ptr()), c_void_p(arg180_1.data_ptr()), c_void_p(arg233_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf173.data_ptr()))
    del arg0_1
    del arg1_1
    del arg233_1
    del buf0
    del buf1
    del buf172
    # Source Nodes: [conv2d, x], Original ATen: [aten.constant_pad_nd, aten.convolution]
    buf174 = extern_kernels.convolution(buf171, buf173, arg2_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf174, (8, 16, 128, 128), (262144, 1, 2048, 16))
    del arg2_1
    del buf171
    del buf173
    buf175 = buf174; del buf174  # reuse
    buf176 = empty((1, 32, 144), device='cpu', dtype=torch.float32)
    buf177 = empty_strided((32, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_convolution_gelu_mul_1(c_void_p(buf175.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf177.data_ptr()))
    del arg3_1
    del arg4_1
    del buf176
    del buf3
    del buf4
    # Source Nodes: [conv2d_1, gelu, mul_], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf178 = extern_kernels.convolution(buf175, buf177, arg5_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf178, (8, 32, 128, 128), (524288, 1, 4096, 32))
    del arg5_1
    del buf175
    del buf177
    buf179 = buf178; del buf178  # reuse
    buf180 = empty((1, 64, 288), device='cpu', dtype=torch.float32)
    buf181 = empty_strided((64, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_convolution_gelu_mul_2(c_void_p(buf179.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(arg7_1.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf181.data_ptr()))
    del arg6_1
    del arg7_1
    del buf180
    del buf6
    del buf7
    # Source Nodes: [conv2d_2, gelu_1, mul__1], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf182 = extern_kernels.convolution(buf179, buf181, arg8_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf182, (8, 64, 128, 128), (1048576, 1, 8192, 64))
    del arg8_1
    del buf179
    del buf181
    buf183 = empty_strided((8, 64, 129, 129), (1065024, 1, 8256, 64), device='cpu', dtype=torch.float32)
    buf184 = empty((1, 128, 576), device='cpu', dtype=torch.float32)
    buf185 = empty_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_3(c_void_p(buf182.data_ptr()), c_void_p(arg9_1.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()))
    del arg10_1
    del arg9_1
    del buf10
    del buf182
    del buf184
    del buf9
    # Source Nodes: [gelu_2, mul__2, shortcut, x_2], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
    buf186 = extern_kernels.convolution(buf183, buf185, arg11_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf186, (8, 128, 64, 64), (524288, 1, 8192, 128))
    del arg11_1
    del buf183
    del buf185
    buf187 = buf186; del buf186  # reuse
    buf188 = empty((1, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_gelu_mul_4(c_void_p(buf187.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(buf188.data_ptr()))
    del arg15_1
    del arg16_1
    del buf15
    del buf16
    # Source Nodes: [gelu_3, mul__3, out, out_1], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf189 = extern_kernels.convolution(buf187, reinterpret_tensor(buf188, (128, 128, 1, 1), (128, 1, 0, 0), 0), arg17_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf189, (8, 128, 64, 64), (524288, 1, 8192, 128))
    del arg17_1
    del buf188
    buf190 = buf189; del buf189  # reuse
    buf191 = empty((1, 128, 1152), device='cpu', dtype=torch.float32)
    buf192 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_convolution_gelu_mul_5(c_void_p(buf190.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf192.data_ptr()))
    del arg18_1
    del arg19_1
    del buf18
    del buf19
    # Source Nodes: [gelu_4, mul__4, out_2], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf193 = extern_kernels.convolution(buf190, buf192, arg20_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf193, (8, 128, 64, 64), (524288, 1, 8192, 128))
    del arg20_1
    del buf190
    buf194 = buf193; del buf193  # reuse
    buf195 = reinterpret_tensor(buf192, (1, 128, 1152), (147456, 1152, 1), 0); del buf192  # reuse
    buf196 = reinterpret_tensor(buf191, (128, 128, 3, 3), (1152, 1, 384, 128), 0); del buf191  # reuse
    cpp_fused__native_batch_norm_legit_convolution_gelu_mul_6(c_void_p(buf194.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf196.data_ptr()))
    del arg21_1
    del arg22_1
    del buf195
    del buf21
    del buf22
    # Source Nodes: [gelu_5, mul__5, out_3], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf197 = extern_kernels.convolution(buf194, buf196, arg23_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf197, (8, 128, 64, 64), (524288, 1, 8192, 128))
    del arg23_1
    del buf194
    del buf196
    buf198 = buf197; del buf197  # reuse
    buf199 = empty((1, 256, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_gelu_mul_7(c_void_p(buf198.data_ptr()), c_void_p(arg24_1.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(arg25_1.data_ptr()), c_void_p(buf199.data_ptr()))
    del arg24_1
    del arg25_1
    del buf24
    del buf25
    # Source Nodes: [gelu_6, mul__6, out_4], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf200 = extern_kernels.convolution(buf198, reinterpret_tensor(buf199, (256, 128, 1, 1), (128, 1, 0, 0), 0), arg26_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf200, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    del arg26_1
    del buf198
    buf201 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cpu', dtype=torch.float32)
    buf202 = reinterpret_tensor(buf201, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf201  # reuse
    cpp_fused_mean_8(c_void_p(buf202.data_ptr()), c_void_p(buf200.data_ptr()))
    # Source Nodes: [x_se, x_se_1], Original ATen: [aten.convolution, aten.mean]
    buf203 = extern_kernels.convolution(buf202, arg183_1, arg184_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf203, (8, 128, 1, 1), (128, 1, 128, 128))
    del arg183_1
    del arg184_1
    del buf202
    buf204 = buf203; del buf203  # reuse
    cpp_fused_relu_9(c_void_p(buf204.data_ptr()))
    # Source Nodes: [x_se_2, x_se_3], Original ATen: [aten.convolution, aten.relu]
    buf205 = extern_kernels.convolution(buf204, arg185_1, arg186_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf205, (8, 256, 1, 1), (256, 1, 256, 256))
    del arg185_1
    del arg186_1
    del buf204
    buf206 = buf199; del buf199  # reuse
    cpp_fused__native_batch_norm_legit_10(c_void_p(arg12_1.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(buf206.data_ptr()))
    del arg12_1
    del arg13_1
    del buf12
    del buf13
    # Source Nodes: [shortcut_1], Original ATen: [aten.convolution]
    buf207 = extern_kernels.convolution(buf187, reinterpret_tensor(buf206, (256, 128, 1, 1), (128, 1, 0, 0), 0), arg14_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf207, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    del arg14_1
    del buf206
    buf208 = buf200; del buf200  # reuse
    buf209 = empty((1, 256, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_add_gelu_mul_sigmoid_11(c_void_p(buf208.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(arg27_1.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(arg31_1.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(buf209.data_ptr()))
    del arg27_1
    del arg31_1
    del arg32_1
    del buf205
    del buf207
    del buf30
    del buf31
    # Source Nodes: [out_9], Original ATen: [aten.convolution]
    buf210 = extern_kernels.convolution(buf208, reinterpret_tensor(buf209, (256, 256, 1, 1), (256, 1, 0, 0), 0), arg33_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf210, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    del arg33_1
    del buf209
    buf211 = empty_strided((8, 256, 65, 65), (1081600, 1, 16640, 256), device='cpu', dtype=torch.float32)
    buf212 = empty((1, 256, 1152), device='cpu', dtype=torch.float32)
    buf213 = empty_strided((256, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_12(c_void_p(buf210.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(arg35_1.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf213.data_ptr()))
    del arg34_1
    del arg35_1
    del buf210
    del buf33
    del buf34
    # Source Nodes: [gelu_8, mul__9, out_10, x_5], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
    buf214 = extern_kernels.convolution(buf211, buf213, arg36_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2)
    assert_size_stride(buf214, (8, 256, 32, 32), (262144, 1, 8192, 256))
    del arg36_1
    del buf211
    buf215 = buf214; del buf214  # reuse
    buf216 = reinterpret_tensor(buf213, (1, 256, 1152), (294912, 1152, 1), 0); del buf213  # reuse
    buf217 = reinterpret_tensor(buf212, (256, 128, 3, 3), (1152, 1, 384, 128), 0); del buf212  # reuse
    cpp_fused__native_batch_norm_legit_convolution_gelu_mul_13(c_void_p(buf215.data_ptr()), c_void_p(arg37_1.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf217.data_ptr()))
    del arg37_1
    del arg38_1
    del buf36
    del buf37
    # Source Nodes: [gelu_9, mul__10, out_11], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf218 = extern_kernels.convolution(buf215, buf217, arg39_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2)
    assert_size_stride(buf218, (8, 256, 32, 32), (262144, 1, 8192, 256))
    del arg39_1
    del buf215
    buf219 = buf218; del buf218  # reuse
    buf220 = empty((1, 512, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_gelu_mul_14(c_void_p(buf219.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(arg41_1.data_ptr()), c_void_p(buf220.data_ptr()))
    del arg40_1
    del arg41_1
    del buf39
    del buf40
    # Source Nodes: [gelu_10, mul__11, out_12], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf221 = extern_kernels.convolution(buf219, reinterpret_tensor(buf220, (512, 256, 1, 1), (256, 1, 0, 0), 0), arg42_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf221, (8, 512, 32, 32), (524288, 1, 16384, 512))
    del arg42_1
    buf222 = empty_strided((8, 512, 1, 1), (512, 1, 4096, 4096), device='cpu', dtype=torch.float32)
    buf223 = reinterpret_tensor(buf222, (8, 512, 1, 1), (512, 1, 512, 512), 0); del buf222  # reuse
    cpp_fused_mean_15(c_void_p(buf223.data_ptr()), c_void_p(buf221.data_ptr()))
    # Source Nodes: [x_se_4, x_se_5], Original ATen: [aten.convolution, aten.mean]
    buf224 = extern_kernels.convolution(buf223, arg187_1, arg188_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf224, (8, 256, 1, 1), (256, 1, 256, 256))
    del arg187_1
    del arg188_1
    buf225 = buf224; del buf224  # reuse
    cpp_fused_relu_16(c_void_p(buf225.data_ptr()))
    # Source Nodes: [x_se_6, x_se_7], Original ATen: [aten.convolution, aten.relu]
    buf226 = extern_kernels.convolution(buf225, arg189_1, arg190_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf226, (8, 512, 1, 1), (512, 1, 512, 512))
    del arg189_1
    del arg190_1
    del buf225
    buf227 = buf219; del buf219  # reuse
    buf228 = buf220; del buf220  # reuse
    cpp_fused__native_batch_norm_legit_avg_pool2d_17(c_void_p(buf208.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(arg29_1.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf228.data_ptr()))
    del arg28_1
    del arg29_1
    del buf208
    del buf27
    del buf28
    # Source Nodes: [getattr_getattr_l__mod___stages___1_____0___downsample_pool, shortcut_3], Original ATen: [aten.avg_pool2d, aten.convolution]
    buf229 = extern_kernels.convolution(buf227, reinterpret_tensor(buf228, (512, 256, 1, 1), (256, 1, 0, 0), 0), arg30_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf229, (8, 512, 32, 32), (524288, 1, 16384, 512))
    del arg30_1
    del buf227
    buf230 = reinterpret_tensor(buf187, (8, 512, 32, 32), (524288, 1, 16384, 512), 0); del buf187  # reuse
    buf231 = reinterpret_tensor(buf228, (1, 256, 512), (131072, 512, 1), 0); del buf228  # reuse
    cpp_fused__native_batch_norm_legit_add_gelu_mul_sigmoid_18(c_void_p(buf221.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(arg43_1.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(arg44_1.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(arg45_1.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()))
    del arg44_1
    del arg45_1
    del buf42
    del buf43
    # Source Nodes: [gelu_11, mul_19, mul_21, mul__12, mul__13, out_13, out_16, out_17, shortcut_4, sigmoid_1], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mul, aten.sigmoid]
    buf232 = extern_kernels.convolution(buf230, reinterpret_tensor(buf231, (256, 512, 1, 1), (512, 1, 0, 0), 0), arg46_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf232, (8, 256, 32, 32), (262144, 1, 8192, 256))
    del arg46_1
    del buf230
    buf233 = buf232; del buf232  # reuse
    buf234 = reinterpret_tensor(buf217, (1, 256, 1152), (294912, 1152, 1), 0); del buf217  # reuse
    buf235 = reinterpret_tensor(buf216, (256, 128, 3, 3), (1152, 1, 384, 128), 0); del buf216  # reuse
    cpp_fused__native_batch_norm_legit_convolution_gelu_mul_19(c_void_p(buf233.data_ptr()), c_void_p(arg47_1.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf235.data_ptr()))
    del arg47_1
    del arg48_1
    del buf45
    del buf46
    # Source Nodes: [gelu_12, mul__14, out_18], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf236 = extern_kernels.convolution(buf233, buf235, arg49_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2)
    assert_size_stride(buf236, (8, 256, 32, 32), (262144, 1, 8192, 256))
    del arg49_1
    del buf233
    buf237 = buf236; del buf236  # reuse
    buf238 = reinterpret_tensor(buf235, (1, 256, 1152), (294912, 1152, 1), 0); del buf235  # reuse
    buf239 = reinterpret_tensor(buf234, (256, 128, 3, 3), (1152, 1, 384, 128), 0); del buf234  # reuse
    cpp_fused__native_batch_norm_legit_convolution_gelu_mul_20(c_void_p(buf237.data_ptr()), c_void_p(arg50_1.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(arg51_1.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf239.data_ptr()))
    del arg50_1
    del arg51_1
    del buf238
    del buf48
    del buf49
    # Source Nodes: [gelu_13, mul__15, out_19], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf240 = extern_kernels.convolution(buf237, buf239, arg52_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2)
    assert_size_stride(buf240, (8, 256, 32, 32), (262144, 1, 8192, 256))
    del arg52_1
    del buf237
    del buf239
    buf241 = buf240; del buf240  # reuse
    buf242 = reinterpret_tensor(buf231, (1, 512, 256), (131072, 256, 1), 0); del buf231  # reuse
    cpp_fused__native_batch_norm_legit_gelu_mul_21(c_void_p(buf241.data_ptr()), c_void_p(arg53_1.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(arg54_1.data_ptr()), c_void_p(buf242.data_ptr()))
    del arg53_1
    del arg54_1
    del buf51
    del buf52
    # Source Nodes: [gelu_14, mul__16, out_20], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf243 = extern_kernels.convolution(buf241, reinterpret_tensor(buf242, (512, 256, 1, 1), (256, 1, 0, 0), 0), arg55_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf243, (8, 512, 32, 32), (524288, 1, 16384, 512))
    del arg55_1
    del buf241
    del buf242
    buf244 = reinterpret_tensor(buf223, (8, 512, 1, 1), (512, 1, 4096, 4096), 0); del buf223  # reuse
    buf245 = reinterpret_tensor(buf244, (8, 512, 1, 1), (512, 1, 512, 512), 0); del buf244  # reuse
    cpp_fused_mean_22(c_void_p(buf245.data_ptr()), c_void_p(buf243.data_ptr()))
    # Source Nodes: [x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean]
    buf246 = extern_kernels.convolution(buf245, arg191_1, arg192_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf246, (8, 256, 1, 1), (256, 1, 256, 256))
    del arg191_1
    del arg192_1
    del buf245
    buf247 = buf246; del buf246  # reuse
    cpp_fused_relu_23(c_void_p(buf247.data_ptr()))
    # Source Nodes: [x_se_10, x_se_11], Original ATen: [aten.convolution, aten.relu]
    buf248 = extern_kernels.convolution(buf247, arg193_1, arg194_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf248, (8, 512, 1, 1), (512, 1, 512, 512))
    del arg193_1
    del arg194_1
    del buf247
    buf249 = buf221; del buf221  # reuse
    buf250 = buf249; del buf249  # reuse
    buf251 = empty((1, 768, 512), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_add_gelu_mul_sigmoid_24(c_void_p(buf250.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(arg56_1.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(arg43_1.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(arg60_1.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(arg61_1.data_ptr()), c_void_p(buf251.data_ptr()))
    del arg43_1
    del arg56_1
    del arg60_1
    del arg61_1
    del buf226
    del buf229
    del buf243
    del buf248
    del buf57
    del buf58
    # Source Nodes: [gelu_15, mul__18, out_24, out_25], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf252 = extern_kernels.convolution(buf250, reinterpret_tensor(buf251, (768, 512, 1, 1), (512, 1, 0, 0), 0), arg62_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf252, (8, 768, 32, 32), (786432, 1, 24576, 768))
    del arg62_1
    del buf251
    buf253 = empty_strided((8, 768, 33, 33), (836352, 1, 25344, 768), device='cpu', dtype=torch.float32)
    buf254 = empty((1, 768, 1152), device='cpu', dtype=torch.float32)
    buf255 = empty_strided((768, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_25(c_void_p(buf252.data_ptr()), c_void_p(arg63_1.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(arg64_1.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf255.data_ptr()))
    del arg63_1
    del arg64_1
    del buf252
    del buf60
    del buf61
    # Source Nodes: [gelu_16, mul__19, out_26, x_7], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
    buf256 = extern_kernels.convolution(buf253, buf255, arg65_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6)
    assert_size_stride(buf256, (8, 768, 16, 16), (196608, 1, 12288, 768))
    del arg65_1
    del buf253
    buf257 = buf256; del buf256  # reuse
    buf258 = reinterpret_tensor(buf255, (1, 768, 1152), (884736, 1152, 1), 0); del buf255  # reuse
    buf259 = reinterpret_tensor(buf254, (768, 128, 3, 3), (1152, 1, 384, 128), 0); del buf254  # reuse
    cpp_fused__native_batch_norm_legit_convolution_gelu_mul_26(c_void_p(buf257.data_ptr()), c_void_p(arg66_1.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(arg67_1.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf259.data_ptr()))
    del arg66_1
    del arg67_1
    del buf63
    del buf64
    # Source Nodes: [gelu_17, mul__20, out_27], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf260 = extern_kernels.convolution(buf257, buf259, arg68_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6)
    assert_size_stride(buf260, (8, 768, 16, 16), (196608, 1, 12288, 768))
    del arg68_1
    del buf257
    buf261 = buf260; del buf260  # reuse
    buf262 = empty((1, 1536, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_gelu_mul_27(c_void_p(buf261.data_ptr()), c_void_p(arg69_1.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(arg70_1.data_ptr()), c_void_p(buf262.data_ptr()))
    del arg69_1
    del arg70_1
    del buf66
    del buf67
    # Source Nodes: [gelu_18, mul__21, out_28], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf263 = extern_kernels.convolution(buf261, reinterpret_tensor(buf262, (1536, 768, 1, 1), (768, 1, 0, 0), 0), arg71_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf263, (8, 1536, 16, 16), (393216, 1, 24576, 1536))
    del arg71_1
    del buf261
    buf264 = empty_strided((8, 1536, 1, 1), (1536, 1, 12288, 12288), device='cpu', dtype=torch.float32)
    buf265 = reinterpret_tensor(buf264, (8, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf264  # reuse
    cpp_fused_mean_28(c_void_p(buf265.data_ptr()), c_void_p(buf263.data_ptr()))
    # Source Nodes: [x_se_12, x_se_13], Original ATen: [aten.convolution, aten.mean]
    buf266 = extern_kernels.convolution(buf265, arg195_1, arg196_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf266, (8, 768, 1, 1), (768, 1, 768, 768))
    del arg195_1
    del arg196_1
    buf267 = buf266; del buf266  # reuse
    cpp_fused_relu_29(c_void_p(buf267.data_ptr()))
    # Source Nodes: [x_se_14, x_se_15], Original ATen: [aten.convolution, aten.relu]
    buf268 = extern_kernels.convolution(buf267, arg197_1, arg198_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf268, (8, 1536, 1, 1), (1536, 1, 1536, 1536))
    del arg197_1
    del arg198_1
    del buf267
    buf269 = empty_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cpu', dtype=torch.float32)
    buf270 = empty((1, 1536, 512), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_avg_pool2d_30(c_void_p(buf250.data_ptr()), c_void_p(arg57_1.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(arg58_1.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf270.data_ptr()))
    del arg57_1
    del arg58_1
    del buf250
    del buf54
    del buf55
    # Source Nodes: [getattr_getattr_l__mod___stages___2_____0___downsample_pool, shortcut_6], Original ATen: [aten.avg_pool2d, aten.convolution]
    buf271 = extern_kernels.convolution(buf269, reinterpret_tensor(buf270, (1536, 512, 1, 1), (512, 1, 0, 0), 0), arg59_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf271, (8, 1536, 16, 16), (393216, 1, 24576, 1536))
    del arg59_1
    del buf269
    buf272 = empty_strided((8, 1536, 16, 16), (393216, 1, 24576, 1536), device='cpu', dtype=torch.float32)
    buf273 = reinterpret_tensor(buf262, (1, 768, 1536), (1179648, 1536, 1), 0); del buf262  # reuse
    cpp_fused__native_batch_norm_legit_add_gelu_mul_sigmoid_31(c_void_p(buf263.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(arg72_1.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(arg73_1.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(arg74_1.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf273.data_ptr()))
    del arg73_1
    del arg74_1
    del buf69
    del buf70
    # Source Nodes: [gelu_19, mul_36, mul_38, mul__22, mul__23, out_29, out_32, out_33, shortcut_7, sigmoid_3], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mul, aten.sigmoid]
    buf274 = extern_kernels.convolution(buf272, reinterpret_tensor(buf273, (768, 1536, 1, 1), (1536, 1, 0, 0), 0), arg75_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf274, (8, 768, 16, 16), (196608, 1, 12288, 768))
    del arg75_1
    buf275 = buf274; del buf274  # reuse
    buf276 = reinterpret_tensor(buf259, (1, 768, 1152), (884736, 1152, 1), 0); del buf259  # reuse
    buf277 = reinterpret_tensor(buf258, (768, 128, 3, 3), (1152, 1, 384, 128), 0); del buf258  # reuse
    cpp_fused__native_batch_norm_legit_convolution_gelu_mul_32(c_void_p(buf275.data_ptr()), c_void_p(arg76_1.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(arg77_1.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf277.data_ptr()))
    del arg76_1
    del arg77_1
    del buf72
    del buf73
    # Source Nodes: [gelu_20, mul__24, out_34], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf278 = extern_kernels.convolution(buf275, buf277, arg78_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6)
    assert_size_stride(buf278, (8, 768, 16, 16), (196608, 1, 12288, 768))
    del arg78_1
    del buf275
    buf279 = buf278; del buf278  # reuse
    buf280 = reinterpret_tensor(buf277, (1, 768, 1152), (884736, 1152, 1), 0); del buf277  # reuse
    buf281 = reinterpret_tensor(buf276, (768, 128, 3, 3), (1152, 1, 384, 128), 0); del buf276  # reuse
    cpp_fused__native_batch_norm_legit_convolution_gelu_mul_33(c_void_p(buf279.data_ptr()), c_void_p(arg79_1.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(arg80_1.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf281.data_ptr()))
    del arg79_1
    del arg80_1
    del buf75
    del buf76
    # Source Nodes: [gelu_21, mul__25, out_35], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf282 = extern_kernels.convolution(buf279, buf281, arg81_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6)
    assert_size_stride(buf282, (8, 768, 16, 16), (196608, 1, 12288, 768))
    del arg81_1
    del buf279
    buf283 = buf282; del buf282  # reuse
    buf284 = reinterpret_tensor(buf273, (1, 1536, 768), (1179648, 768, 1), 0); del buf273  # reuse
    cpp_fused__native_batch_norm_legit_gelu_mul_34(c_void_p(buf283.data_ptr()), c_void_p(arg82_1.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(arg83_1.data_ptr()), c_void_p(buf284.data_ptr()))
    del arg82_1
    del arg83_1
    del buf78
    del buf79
    # Source Nodes: [gelu_22, mul__26, out_36], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf285 = extern_kernels.convolution(buf283, reinterpret_tensor(buf284, (1536, 768, 1, 1), (768, 1, 0, 0), 0), arg84_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf285, (8, 1536, 16, 16), (393216, 1, 24576, 1536))
    del arg84_1
    del buf283
    buf286 = reinterpret_tensor(buf265, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf265  # reuse
    buf287 = reinterpret_tensor(buf286, (8, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf286  # reuse
    cpp_fused_mean_35(c_void_p(buf287.data_ptr()), c_void_p(buf285.data_ptr()))
    # Source Nodes: [x_se_16, x_se_17], Original ATen: [aten.convolution, aten.mean]
    buf288 = extern_kernels.convolution(buf287, arg199_1, arg200_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf288, (8, 768, 1, 1), (768, 1, 768, 768))
    del arg199_1
    del arg200_1
    del buf287
    buf289 = buf288; del buf288  # reuse
    cpp_fused_relu_36(c_void_p(buf289.data_ptr()))
    # Source Nodes: [x_se_18, x_se_19], Original ATen: [aten.convolution, aten.relu]
    buf290 = extern_kernels.convolution(buf289, arg201_1, arg202_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf290, (8, 1536, 1, 1), (1536, 1, 1536, 1536))
    del arg201_1
    del arg202_1
    del buf289
    buf291 = buf263; del buf263  # reuse
    buf292 = buf272; del buf272  # reuse
    buf293 = reinterpret_tensor(buf284, (1, 768, 1536), (1179648, 1536, 1), 0); del buf284  # reuse
    cpp_fused__native_batch_norm_legit_add_gelu_mul_sigmoid_37(c_void_p(buf291.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(arg85_1.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(arg72_1.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(arg86_1.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(arg87_1.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf293.data_ptr()))
    del arg72_1
    del arg85_1
    del arg86_1
    del arg87_1
    del buf268
    del buf271
    del buf285
    del buf81
    del buf82
    # Source Nodes: [gelu_23, mul__28, out_40, out_41], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf294 = extern_kernels.convolution(buf292, reinterpret_tensor(buf293, (768, 1536, 1, 1), (1536, 1, 0, 0), 0), arg88_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf294, (8, 768, 16, 16), (196608, 1, 12288, 768))
    del arg88_1
    buf295 = buf294; del buf294  # reuse
    buf296 = reinterpret_tensor(buf281, (1, 768, 1152), (884736, 1152, 1), 0); del buf281  # reuse
    buf297 = reinterpret_tensor(buf280, (768, 128, 3, 3), (1152, 1, 384, 128), 0); del buf280  # reuse
    cpp_fused__native_batch_norm_legit_convolution_gelu_mul_38(c_void_p(buf295.data_ptr()), c_void_p(arg89_1.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(arg90_1.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf297.data_ptr()))
    del arg89_1
    del arg90_1
    del buf84
    del buf85
    # Source Nodes: [gelu_24, mul__29, out_42], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf298 = extern_kernels.convolution(buf295, buf297, arg91_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6)
    assert_size_stride(buf298, (8, 768, 16, 16), (196608, 1, 12288, 768))
    del arg91_1
    del buf295
    buf299 = buf298; del buf298  # reuse
    buf300 = reinterpret_tensor(buf297, (1, 768, 1152), (884736, 1152, 1), 0); del buf297  # reuse
    buf301 = reinterpret_tensor(buf296, (768, 128, 3, 3), (1152, 1, 384, 128), 0); del buf296  # reuse
    cpp_fused__native_batch_norm_legit_convolution_gelu_mul_39(c_void_p(buf299.data_ptr()), c_void_p(arg92_1.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(arg93_1.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf301.data_ptr()))
    del arg92_1
    del arg93_1
    del buf87
    del buf88
    # Source Nodes: [gelu_25, mul__30, out_43], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf302 = extern_kernels.convolution(buf299, buf301, arg94_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6)
    assert_size_stride(buf302, (8, 768, 16, 16), (196608, 1, 12288, 768))
    del arg94_1
    del buf299
    buf303 = buf302; del buf302  # reuse
    buf304 = reinterpret_tensor(buf293, (1, 1536, 768), (1179648, 768, 1), 0); del buf293  # reuse
    cpp_fused__native_batch_norm_legit_gelu_mul_40(c_void_p(buf303.data_ptr()), c_void_p(arg95_1.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(arg96_1.data_ptr()), c_void_p(buf304.data_ptr()))
    del arg95_1
    del arg96_1
    del buf90
    del buf91
    # Source Nodes: [gelu_26, mul__31, out_44], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf305 = extern_kernels.convolution(buf303, reinterpret_tensor(buf304, (1536, 768, 1, 1), (768, 1, 0, 0), 0), arg97_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf305, (8, 1536, 16, 16), (393216, 1, 24576, 1536))
    del arg97_1
    del buf303
    buf306 = reinterpret_tensor(buf290, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf290  # reuse
    buf307 = reinterpret_tensor(buf306, (8, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf306  # reuse
    cpp_fused_mean_41(c_void_p(buf307.data_ptr()), c_void_p(buf305.data_ptr()))
    # Source Nodes: [x_se_20, x_se_21], Original ATen: [aten.convolution, aten.mean]
    buf308 = extern_kernels.convolution(buf307, arg203_1, arg204_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf308, (8, 768, 1, 1), (768, 1, 768, 768))
    del arg203_1
    del arg204_1
    buf309 = buf308; del buf308  # reuse
    cpp_fused_relu_42(c_void_p(buf309.data_ptr()))
    # Source Nodes: [x_se_22, x_se_23], Original ATen: [aten.convolution, aten.relu]
    buf310 = extern_kernels.convolution(buf309, arg205_1, arg206_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf310, (8, 1536, 1, 1), (1536, 1, 1536, 1536))
    del arg205_1
    del arg206_1
    del buf309
    buf311 = buf292; del buf292  # reuse
    buf312 = reinterpret_tensor(buf304, (1, 768, 1536), (1179648, 1536, 1), 0); del buf304  # reuse
    cpp_fused__native_batch_norm_legit_add_gelu_mul_sigmoid_43(c_void_p(buf305.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(arg98_1.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(arg99_1.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(arg100_1.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf312.data_ptr()))
    del arg100_1
    del arg99_1
    del buf93
    del buf94
    # Source Nodes: [gelu_27, mul_52, mul_54, mul__32, mul__33, out_45, out_48, out_49, shortcut_9, sigmoid_5], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mul, aten.sigmoid]
    buf313 = extern_kernels.convolution(buf311, reinterpret_tensor(buf312, (768, 1536, 1, 1), (1536, 1, 0, 0), 0), arg101_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf313, (8, 768, 16, 16), (196608, 1, 12288, 768))
    del arg101_1
    buf314 = buf313; del buf313  # reuse
    buf315 = reinterpret_tensor(buf301, (1, 768, 1152), (884736, 1152, 1), 0); del buf301  # reuse
    buf316 = reinterpret_tensor(buf300, (768, 128, 3, 3), (1152, 1, 384, 128), 0); del buf300  # reuse
    cpp_fused__native_batch_norm_legit_convolution_gelu_mul_44(c_void_p(buf314.data_ptr()), c_void_p(arg102_1.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(arg103_1.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf316.data_ptr()))
    del arg102_1
    del arg103_1
    del buf96
    del buf97
    # Source Nodes: [gelu_28, mul__34, out_50], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf317 = extern_kernels.convolution(buf314, buf316, arg104_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6)
    assert_size_stride(buf317, (8, 768, 16, 16), (196608, 1, 12288, 768))
    del arg104_1
    del buf314
    buf318 = buf317; del buf317  # reuse
    buf319 = reinterpret_tensor(buf316, (1, 768, 1152), (884736, 1152, 1), 0); del buf316  # reuse
    buf320 = reinterpret_tensor(buf315, (768, 128, 3, 3), (1152, 1, 384, 128), 0); del buf315  # reuse
    cpp_fused__native_batch_norm_legit_convolution_gelu_mul_45(c_void_p(buf318.data_ptr()), c_void_p(arg105_1.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(arg106_1.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf320.data_ptr()))
    del arg105_1
    del arg106_1
    del buf100
    del buf99
    # Source Nodes: [gelu_29, mul__35, out_51], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf321 = extern_kernels.convolution(buf318, buf320, arg107_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6)
    assert_size_stride(buf321, (8, 768, 16, 16), (196608, 1, 12288, 768))
    del arg107_1
    del buf318
    buf322 = buf321; del buf321  # reuse
    buf323 = reinterpret_tensor(buf312, (1, 1536, 768), (1179648, 768, 1), 0); del buf312  # reuse
    cpp_fused__native_batch_norm_legit_gelu_mul_46(c_void_p(buf322.data_ptr()), c_void_p(arg108_1.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(arg109_1.data_ptr()), c_void_p(buf323.data_ptr()))
    del arg108_1
    del arg109_1
    del buf102
    del buf103
    # Source Nodes: [gelu_30, mul__36, out_52], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf324 = extern_kernels.convolution(buf322, reinterpret_tensor(buf323, (1536, 768, 1, 1), (768, 1, 0, 0), 0), arg110_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf324, (8, 1536, 16, 16), (393216, 1, 24576, 1536))
    del arg110_1
    del buf322
    buf325 = reinterpret_tensor(buf307, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf307  # reuse
    buf326 = reinterpret_tensor(buf325, (8, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf325  # reuse
    cpp_fused_mean_47(c_void_p(buf326.data_ptr()), c_void_p(buf324.data_ptr()))
    # Source Nodes: [x_se_24, x_se_25], Original ATen: [aten.convolution, aten.mean]
    buf327 = extern_kernels.convolution(buf326, arg207_1, arg208_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf327, (8, 768, 1, 1), (768, 1, 768, 768))
    del arg207_1
    del arg208_1
    del buf326
    buf328 = buf327; del buf327  # reuse
    cpp_fused_relu_48(c_void_p(buf328.data_ptr()))
    # Source Nodes: [x_se_26, x_se_27], Original ATen: [aten.convolution, aten.relu]
    buf329 = extern_kernels.convolution(buf328, arg209_1, arg210_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf329, (8, 1536, 1, 1), (1536, 1, 1536, 1536))
    del arg209_1
    del arg210_1
    del buf328
    buf330 = buf291; del buf291  # reuse
    buf331 = buf311; del buf311  # reuse
    buf332 = reinterpret_tensor(buf323, (1, 768, 1536), (1179648, 1536, 1), 0); del buf323  # reuse
    cpp_fused__native_batch_norm_legit_add_gelu_mul_sigmoid_49(c_void_p(buf330.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(arg111_1.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(arg98_1.data_ptr()), c_void_p(arg112_1.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(arg113_1.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf332.data_ptr()))
    del arg111_1
    del arg112_1
    del arg113_1
    del arg98_1
    del buf105
    del buf106
    del buf305
    del buf310
    del buf324
    # Source Nodes: [gelu_31, mul__38, out_56, out_57], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf333 = extern_kernels.convolution(buf331, reinterpret_tensor(buf332, (768, 1536, 1, 1), (1536, 1, 0, 0), 0), arg114_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf333, (8, 768, 16, 16), (196608, 1, 12288, 768))
    del arg114_1
    buf334 = buf333; del buf333  # reuse
    buf335 = reinterpret_tensor(buf320, (1, 768, 1152), (884736, 1152, 1), 0); del buf320  # reuse
    buf336 = reinterpret_tensor(buf319, (768, 128, 3, 3), (1152, 1, 384, 128), 0); del buf319  # reuse
    cpp_fused__native_batch_norm_legit_convolution_gelu_mul_50(c_void_p(buf334.data_ptr()), c_void_p(arg115_1.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(arg116_1.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf336.data_ptr()))
    del arg115_1
    del arg116_1
    del buf108
    del buf109
    # Source Nodes: [gelu_32, mul__39, out_58], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf337 = extern_kernels.convolution(buf334, buf336, arg117_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6)
    assert_size_stride(buf337, (8, 768, 16, 16), (196608, 1, 12288, 768))
    del arg117_1
    del buf334
    buf338 = buf337; del buf337  # reuse
    buf339 = reinterpret_tensor(buf336, (1, 768, 1152), (884736, 1152, 1), 0); del buf336  # reuse
    buf340 = reinterpret_tensor(buf335, (768, 128, 3, 3), (1152, 1, 384, 128), 0); del buf335  # reuse
    cpp_fused__native_batch_norm_legit_convolution_gelu_mul_51(c_void_p(buf338.data_ptr()), c_void_p(arg118_1.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(arg119_1.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf340.data_ptr()))
    del arg118_1
    del arg119_1
    del buf111
    del buf112
    # Source Nodes: [gelu_33, mul__40, out_59], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf341 = extern_kernels.convolution(buf338, buf340, arg120_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6)
    assert_size_stride(buf341, (8, 768, 16, 16), (196608, 1, 12288, 768))
    del arg120_1
    del buf338
    buf342 = buf341; del buf341  # reuse
    buf343 = reinterpret_tensor(buf332, (1, 1536, 768), (1179648, 768, 1), 0); del buf332  # reuse
    cpp_fused__native_batch_norm_legit_gelu_mul_52(c_void_p(buf342.data_ptr()), c_void_p(arg121_1.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(arg122_1.data_ptr()), c_void_p(buf343.data_ptr()))
    del arg121_1
    del arg122_1
    del buf114
    del buf115
    # Source Nodes: [gelu_34, mul__41, out_60], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf344 = extern_kernels.convolution(buf342, reinterpret_tensor(buf343, (1536, 768, 1, 1), (768, 1, 0, 0), 0), arg123_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf344, (8, 1536, 16, 16), (393216, 1, 24576, 1536))
    del arg123_1
    del buf342
    buf345 = reinterpret_tensor(buf329, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf329  # reuse
    buf346 = reinterpret_tensor(buf345, (8, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf345  # reuse
    cpp_fused_mean_53(c_void_p(buf346.data_ptr()), c_void_p(buf344.data_ptr()))
    # Source Nodes: [x_se_28, x_se_29], Original ATen: [aten.convolution, aten.mean]
    buf347 = extern_kernels.convolution(buf346, arg211_1, arg212_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf347, (8, 768, 1, 1), (768, 1, 768, 768))
    del arg211_1
    del arg212_1
    buf348 = buf347; del buf347  # reuse
    cpp_fused_relu_54(c_void_p(buf348.data_ptr()))
    # Source Nodes: [x_se_30, x_se_31], Original ATen: [aten.convolution, aten.relu]
    buf349 = extern_kernels.convolution(buf348, arg213_1, arg214_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf349, (8, 1536, 1, 1), (1536, 1, 1536, 1536))
    del arg213_1
    del arg214_1
    del buf348
    buf350 = buf331; del buf331  # reuse
    buf351 = reinterpret_tensor(buf343, (1, 768, 1536), (1179648, 1536, 1), 0); del buf343  # reuse
    cpp_fused__native_batch_norm_legit_add_gelu_mul_sigmoid_55(c_void_p(buf344.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(arg124_1.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(arg125_1.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(arg126_1.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf351.data_ptr()))
    del arg125_1
    del arg126_1
    del buf117
    del buf118
    # Source Nodes: [gelu_35, mul_68, mul_70, mul__42, mul__43, out_61, out_64, out_65, shortcut_11, sigmoid_7], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mul, aten.sigmoid]
    buf352 = extern_kernels.convolution(buf350, reinterpret_tensor(buf351, (768, 1536, 1, 1), (1536, 1, 0, 0), 0), arg127_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf352, (8, 768, 16, 16), (196608, 1, 12288, 768))
    del arg127_1
    del buf350
    buf353 = buf352; del buf352  # reuse
    buf354 = reinterpret_tensor(buf340, (1, 768, 1152), (884736, 1152, 1), 0); del buf340  # reuse
    buf355 = reinterpret_tensor(buf339, (768, 128, 3, 3), (1152, 1, 384, 128), 0); del buf339  # reuse
    cpp_fused__native_batch_norm_legit_convolution_gelu_mul_56(c_void_p(buf353.data_ptr()), c_void_p(arg128_1.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(arg129_1.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf355.data_ptr()))
    del arg128_1
    del arg129_1
    del buf120
    del buf121
    # Source Nodes: [gelu_36, mul__44, out_66], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf356 = extern_kernels.convolution(buf353, buf355, arg130_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6)
    assert_size_stride(buf356, (8, 768, 16, 16), (196608, 1, 12288, 768))
    del arg130_1
    del buf353
    buf357 = buf356; del buf356  # reuse
    buf358 = reinterpret_tensor(buf355, (1, 768, 1152), (884736, 1152, 1), 0); del buf355  # reuse
    buf359 = reinterpret_tensor(buf354, (768, 128, 3, 3), (1152, 1, 384, 128), 0); del buf354  # reuse
    cpp_fused__native_batch_norm_legit_convolution_gelu_mul_57(c_void_p(buf357.data_ptr()), c_void_p(arg131_1.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(arg132_1.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf359.data_ptr()))
    del arg131_1
    del arg132_1
    del buf123
    del buf124
    # Source Nodes: [gelu_37, mul__45, out_67], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf360 = extern_kernels.convolution(buf357, buf359, arg133_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6)
    assert_size_stride(buf360, (8, 768, 16, 16), (196608, 1, 12288, 768))
    del arg133_1
    del buf357
    buf361 = buf360; del buf360  # reuse
    buf362 = reinterpret_tensor(buf351, (1, 1536, 768), (1179648, 768, 1), 0); del buf351  # reuse
    cpp_fused__native_batch_norm_legit_gelu_mul_58(c_void_p(buf361.data_ptr()), c_void_p(arg134_1.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(arg135_1.data_ptr()), c_void_p(buf362.data_ptr()))
    del arg134_1
    del arg135_1
    del buf126
    del buf127
    # Source Nodes: [gelu_38, mul__46, out_68], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf363 = extern_kernels.convolution(buf361, reinterpret_tensor(buf362, (1536, 768, 1, 1), (768, 1, 0, 0), 0), arg136_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf363, (8, 1536, 16, 16), (393216, 1, 24576, 1536))
    del arg136_1
    del buf361
    buf364 = reinterpret_tensor(buf346, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf346  # reuse
    buf365 = reinterpret_tensor(buf364, (8, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf364  # reuse
    cpp_fused_mean_59(c_void_p(buf365.data_ptr()), c_void_p(buf363.data_ptr()))
    # Source Nodes: [x_se_32, x_se_33], Original ATen: [aten.convolution, aten.mean]
    buf366 = extern_kernels.convolution(buf365, arg215_1, arg216_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf366, (8, 768, 1, 1), (768, 1, 768, 768))
    del arg215_1
    del arg216_1
    del buf365
    buf367 = buf366; del buf366  # reuse
    cpp_fused_relu_60(c_void_p(buf367.data_ptr()))
    # Source Nodes: [x_se_34, x_se_35], Original ATen: [aten.convolution, aten.relu]
    buf368 = extern_kernels.convolution(buf367, arg217_1, arg218_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf368, (8, 1536, 1, 1), (1536, 1, 1536, 1536))
    del arg217_1
    del arg218_1
    del buf367
    buf369 = buf330; del buf330  # reuse
    buf370 = buf369; del buf369  # reuse
    buf371 = reinterpret_tensor(buf362, (1, 768, 1536), (1179648, 1536, 1), 0); del buf362  # reuse
    cpp_fused__native_batch_norm_legit_add_gelu_mul_sigmoid_61(c_void_p(buf370.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(arg137_1.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(arg124_1.data_ptr()), c_void_p(arg141_1.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(arg142_1.data_ptr()), c_void_p(buf371.data_ptr()))
    del arg124_1
    del arg137_1
    del arg141_1
    del arg142_1
    del buf132
    del buf133
    del buf344
    del buf349
    del buf363
    # Source Nodes: [gelu_39, mul__48, out_72, out_73], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf372 = extern_kernels.convolution(buf370, reinterpret_tensor(buf371, (768, 1536, 1, 1), (1536, 1, 0, 0), 0), arg143_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf372, (8, 768, 16, 16), (196608, 1, 12288, 768))
    del arg143_1
    buf373 = empty_strided((8, 768, 17, 17), (221952, 1, 13056, 768), device='cpu', dtype=torch.float32)
    buf374 = reinterpret_tensor(buf359, (1, 768, 1152), (884736, 1152, 1), 0); del buf359  # reuse
    buf375 = reinterpret_tensor(buf358, (768, 128, 3, 3), (1152, 1, 384, 128), 0); del buf358  # reuse
    cpp_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_62(c_void_p(buf372.data_ptr()), c_void_p(arg144_1.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(arg145_1.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf375.data_ptr()))
    del arg144_1
    del arg145_1
    del buf135
    del buf136
    del buf372
    # Source Nodes: [gelu_40, mul__49, out_74, x_9], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
    buf376 = extern_kernels.convolution(buf373, buf375, arg146_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6)
    assert_size_stride(buf376, (8, 768, 8, 8), (49152, 1, 6144, 768))
    del arg146_1
    del buf373
    buf377 = buf376; del buf376  # reuse
    buf378 = reinterpret_tensor(buf375, (1, 768, 1152), (884736, 1152, 1), 0); del buf375  # reuse
    buf379 = reinterpret_tensor(buf374, (768, 128, 3, 3), (1152, 1, 384, 128), 0); del buf374  # reuse
    cpp_fused__native_batch_norm_legit_convolution_gelu_mul_63(c_void_p(buf377.data_ptr()), c_void_p(arg147_1.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(arg148_1.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf379.data_ptr()))
    del arg147_1
    del arg148_1
    del buf138
    del buf139
    # Source Nodes: [gelu_41, mul__50, out_75], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf380 = extern_kernels.convolution(buf377, buf379, arg149_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6)
    assert_size_stride(buf380, (8, 768, 8, 8), (49152, 1, 6144, 768))
    del arg149_1
    del buf377
    buf381 = buf380; del buf380  # reuse
    buf382 = reinterpret_tensor(buf371, (1, 1536, 768), (1179648, 768, 1), 0); del buf371  # reuse
    cpp_fused__native_batch_norm_legit_gelu_mul_64(c_void_p(buf381.data_ptr()), c_void_p(arg150_1.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(arg151_1.data_ptr()), c_void_p(buf382.data_ptr()))
    del arg150_1
    del arg151_1
    del buf141
    del buf142
    # Source Nodes: [gelu_42, mul__51, out_76], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf383 = extern_kernels.convolution(buf381, reinterpret_tensor(buf382, (1536, 768, 1, 1), (768, 1, 0, 0), 0), arg152_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf383, (8, 1536, 8, 8), (98304, 1, 12288, 1536))
    del arg152_1
    del buf381
    buf384 = reinterpret_tensor(buf368, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf368  # reuse
    buf385 = reinterpret_tensor(buf384, (8, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf384  # reuse
    cpp_fused_mean_65(c_void_p(buf385.data_ptr()), c_void_p(buf383.data_ptr()))
    # Source Nodes: [x_se_36, x_se_37], Original ATen: [aten.convolution, aten.mean]
    buf386 = extern_kernels.convolution(buf385, arg219_1, arg220_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf386, (8, 768, 1, 1), (768, 1, 768, 768))
    del arg219_1
    del arg220_1
    buf387 = buf386; del buf386  # reuse
    cpp_fused_relu_66(c_void_p(buf387.data_ptr()))
    # Source Nodes: [x_se_38, x_se_39], Original ATen: [aten.convolution, aten.relu]
    buf388 = extern_kernels.convolution(buf387, arg221_1, arg222_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf388, (8, 1536, 1, 1), (1536, 1, 1536, 1536))
    del arg221_1
    del arg222_1
    del buf387
    buf389 = reinterpret_tensor(buf270, (8, 1536, 8, 8), (98304, 1, 12288, 1536), 0); del buf270  # reuse
    buf390 = empty((1, 1536, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_avg_pool2d_67(c_void_p(buf370.data_ptr()), c_void_p(arg138_1.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(arg139_1.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf390.data_ptr()))
    del arg138_1
    del arg139_1
    del buf129
    del buf130
    del buf370
    # Source Nodes: [getattr_getattr_l__mod___stages___3_____0___downsample_pool, shortcut_13], Original ATen: [aten.avg_pool2d, aten.convolution]
    buf391 = extern_kernels.convolution(buf389, reinterpret_tensor(buf390, (1536, 1536, 1, 1), (1536, 1, 0, 0), 0), arg140_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf391, (8, 1536, 8, 8), (98304, 1, 12288, 1536))
    del arg140_1
    del buf390
    buf392 = buf389; del buf389  # reuse
    buf393 = reinterpret_tensor(buf382, (1, 768, 1536), (1179648, 1536, 1), 0); del buf382  # reuse
    cpp_fused__native_batch_norm_legit_add_gelu_mul_sigmoid_68(c_void_p(buf383.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(arg153_1.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(arg154_1.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(arg155_1.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(buf393.data_ptr()))
    del arg154_1
    del arg155_1
    del buf144
    del buf145
    # Source Nodes: [gelu_43, mul_85, mul_87, mul__52, mul__53, out_77, out_80, out_81, shortcut_14, sigmoid_9], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mul, aten.sigmoid]
    buf394 = extern_kernels.convolution(buf392, reinterpret_tensor(buf393, (768, 1536, 1, 1), (1536, 1, 0, 0), 0), arg156_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf394, (8, 768, 8, 8), (49152, 1, 6144, 768))
    del arg156_1
    buf395 = buf394; del buf394  # reuse
    buf396 = reinterpret_tensor(buf379, (1, 768, 1152), (884736, 1152, 1), 0); del buf379  # reuse
    buf397 = reinterpret_tensor(buf378, (768, 128, 3, 3), (1152, 1, 384, 128), 0); del buf378  # reuse
    cpp_fused__native_batch_norm_legit_convolution_gelu_mul_69(c_void_p(buf395.data_ptr()), c_void_p(arg157_1.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(arg158_1.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf397.data_ptr()))
    del arg157_1
    del arg158_1
    del buf147
    del buf148
    # Source Nodes: [gelu_44, mul__54, out_82], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf398 = extern_kernels.convolution(buf395, buf397, arg159_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6)
    assert_size_stride(buf398, (8, 768, 8, 8), (49152, 1, 6144, 768))
    del arg159_1
    del buf395
    buf399 = buf398; del buf398  # reuse
    buf400 = reinterpret_tensor(buf397, (1, 768, 1152), (884736, 1152, 1), 0); del buf397  # reuse
    buf401 = reinterpret_tensor(buf396, (768, 128, 3, 3), (1152, 1, 384, 128), 0); del buf396  # reuse
    cpp_fused__native_batch_norm_legit_convolution_gelu_mul_70(c_void_p(buf399.data_ptr()), c_void_p(arg160_1.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(arg161_1.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(buf401.data_ptr()))
    del arg160_1
    del arg161_1
    del buf150
    del buf151
    # Source Nodes: [gelu_45, mul__55, out_83], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf402 = extern_kernels.convolution(buf399, buf401, arg162_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6)
    assert_size_stride(buf402, (8, 768, 8, 8), (49152, 1, 6144, 768))
    del arg162_1
    del buf399
    buf403 = buf402; del buf402  # reuse
    buf404 = reinterpret_tensor(buf393, (1, 1536, 768), (1179648, 768, 1), 0); del buf393  # reuse
    cpp_fused__native_batch_norm_legit_gelu_mul_71(c_void_p(buf403.data_ptr()), c_void_p(arg163_1.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(arg164_1.data_ptr()), c_void_p(buf404.data_ptr()))
    del arg163_1
    del arg164_1
    del buf153
    del buf154
    # Source Nodes: [gelu_46, mul__56, out_84], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf405 = extern_kernels.convolution(buf403, reinterpret_tensor(buf404, (1536, 768, 1, 1), (768, 1, 0, 0), 0), arg165_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf405, (8, 1536, 8, 8), (98304, 1, 12288, 1536))
    del arg165_1
    del buf403
    buf406 = reinterpret_tensor(buf385, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf385  # reuse
    buf407 = reinterpret_tensor(buf406, (8, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf406  # reuse
    cpp_fused_mean_72(c_void_p(buf407.data_ptr()), c_void_p(buf405.data_ptr()))
    # Source Nodes: [x_se_40, x_se_41], Original ATen: [aten.convolution, aten.mean]
    buf408 = extern_kernels.convolution(buf407, arg223_1, arg224_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf408, (8, 768, 1, 1), (768, 1, 768, 768))
    del arg223_1
    del arg224_1
    del buf407
    buf409 = buf408; del buf408  # reuse
    cpp_fused_relu_73(c_void_p(buf409.data_ptr()))
    # Source Nodes: [x_se_42, x_se_43], Original ATen: [aten.convolution, aten.relu]
    buf410 = extern_kernels.convolution(buf409, arg225_1, arg226_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf410, (8, 1536, 1, 1), (1536, 1, 1536, 1536))
    del arg225_1
    del arg226_1
    del buf409
    buf411 = buf383; del buf383  # reuse
    buf412 = buf392; del buf392  # reuse
    buf413 = reinterpret_tensor(buf404, (1, 768, 1536), (1179648, 1536, 1), 0); del buf404  # reuse
    cpp_fused__native_batch_norm_legit_add_gelu_mul_sigmoid_74(c_void_p(buf411.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(arg166_1.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(arg153_1.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(arg167_1.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(arg168_1.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf413.data_ptr()))
    del arg153_1
    del arg166_1
    del arg167_1
    del arg168_1
    del buf156
    del buf157
    del buf388
    del buf391
    del buf405
    # Source Nodes: [gelu_47, mul__58, out_88, out_89], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf414 = extern_kernels.convolution(buf412, reinterpret_tensor(buf413, (768, 1536, 1, 1), (1536, 1, 0, 0), 0), arg169_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf414, (8, 768, 8, 8), (49152, 1, 6144, 768))
    del arg169_1
    del buf412
    buf415 = buf414; del buf414  # reuse
    buf416 = reinterpret_tensor(buf401, (1, 768, 1152), (884736, 1152, 1), 0); del buf401  # reuse
    buf417 = reinterpret_tensor(buf400, (768, 128, 3, 3), (1152, 1, 384, 128), 0); del buf400  # reuse
    cpp_fused__native_batch_norm_legit_convolution_gelu_mul_75(c_void_p(buf415.data_ptr()), c_void_p(arg170_1.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(arg171_1.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(buf417.data_ptr()))
    del arg170_1
    del arg171_1
    del buf159
    del buf160
    # Source Nodes: [gelu_48, mul__59, out_90], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf418 = extern_kernels.convolution(buf415, buf417, arg172_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6)
    assert_size_stride(buf418, (8, 768, 8, 8), (49152, 1, 6144, 768))
    del arg172_1
    del buf415
    buf419 = buf418; del buf418  # reuse
    buf420 = reinterpret_tensor(buf417, (1, 768, 1152), (884736, 1152, 1), 0); del buf417  # reuse
    buf421 = reinterpret_tensor(buf416, (768, 128, 3, 3), (1152, 1, 384, 128), 0); del buf416  # reuse
    cpp_fused__native_batch_norm_legit_convolution_gelu_mul_76(c_void_p(buf419.data_ptr()), c_void_p(arg173_1.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(arg174_1.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(buf421.data_ptr()))
    del arg173_1
    del arg174_1
    del buf162
    del buf163
    del buf420
    # Source Nodes: [gelu_49, mul__60, out_91], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf422 = extern_kernels.convolution(buf419, buf421, arg175_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6)
    assert_size_stride(buf422, (8, 768, 8, 8), (49152, 1, 6144, 768))
    del arg175_1
    del buf419
    del buf421
    buf423 = buf422; del buf422  # reuse
    buf424 = reinterpret_tensor(buf413, (1, 1536, 768), (1179648, 768, 1), 0); del buf413  # reuse
    cpp_fused__native_batch_norm_legit_gelu_mul_77(c_void_p(buf423.data_ptr()), c_void_p(arg176_1.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(arg177_1.data_ptr()), c_void_p(buf424.data_ptr()))
    del arg176_1
    del arg177_1
    del buf165
    del buf166
    # Source Nodes: [gelu_50, mul__61, out_92], Original ATen: [aten.convolution, aten.gelu, aten.mul]
    buf425 = extern_kernels.convolution(buf423, reinterpret_tensor(buf424, (1536, 768, 1, 1), (768, 1, 0, 0), 0), arg178_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf425, (8, 1536, 8, 8), (98304, 1, 12288, 1536))
    del arg178_1
    del buf423
    del buf424
    buf426 = reinterpret_tensor(buf410, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf410  # reuse
    buf427 = reinterpret_tensor(buf426, (8, 1536, 1, 1), (1536, 1, 1536, 1536), 0); del buf426  # reuse
    cpp_fused_mean_78(c_void_p(buf427.data_ptr()), c_void_p(buf425.data_ptr()))
    # Source Nodes: [x_se_44, x_se_45], Original ATen: [aten.convolution, aten.mean]
    buf428 = extern_kernels.convolution(buf427, arg227_1, arg228_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf428, (8, 768, 1, 1), (768, 1, 768, 768))
    del arg227_1
    del arg228_1
    del buf427
    buf429 = buf428; del buf428  # reuse
    cpp_fused_relu_79(c_void_p(buf429.data_ptr()))
    # Source Nodes: [x_se_46, x_se_47], Original ATen: [aten.convolution, aten.relu]
    buf430 = extern_kernels.convolution(buf429, arg229_1, arg230_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf430, (8, 1536, 1, 1), (1536, 1, 1536, 1536))
    del arg229_1
    del arg230_1
    del buf429
    buf431 = buf411; del buf411  # reuse
    buf432 = empty((1, 3072, 1536), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_add_mul_sigmoid_80(c_void_p(buf431.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(arg179_1.data_ptr()), c_void_p(arg180_1.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(arg181_1.data_ptr()), c_void_p(buf432.data_ptr()))
    del arg179_1
    del arg180_1
    del arg181_1
    del buf168
    del buf169
    del buf425
    del buf430
    # Source Nodes: [mul_101, mul_103, mul__62, out_93, sigmoid_11, x_10, x_11], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sigmoid]
    buf433 = extern_kernels.convolution(buf431, reinterpret_tensor(buf432, (3072, 1536, 1, 1), (1536, 1, 0, 0), 0), arg182_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf433, (8, 3072, 8, 8), (196608, 1, 24576, 3072))
    del arg182_1
    del buf431
    del buf432
    buf434 = empty_strided((8, 3072, 1, 1), (3072, 1, 24576, 24576), device='cpu', dtype=torch.float32)
    buf435 = reinterpret_tensor(buf434, (8, 3072, 1, 1), (3072, 1, 1, 1), 0); del buf434  # reuse
    cpp_fused_gelu_mean_mul_81(c_void_p(buf435.data_ptr()), c_void_p(buf433.data_ptr()))
    del buf433
    buf436 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_18], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg232_1, reinterpret_tensor(buf435, (8, 3072), (3072, 1), 0), reinterpret_tensor(arg231_1, (3072, 1000), (1, 3072), 0), alpha=1, beta=1, out=buf436)
    del arg231_1
    del arg232_1
    return (buf436, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((16, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((32, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((32, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((64, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((1536, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((768, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((1536, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((3072, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((3072, 1, 1, 1), (1, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((1000, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((8, 3, 256, 256), (196608, 65536, 256, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('dm_nfnet_f0', benchmark_compiled_module)
