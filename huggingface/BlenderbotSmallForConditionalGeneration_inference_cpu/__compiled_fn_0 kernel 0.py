
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


cpp_fused_add_arange_embedding_mul_native_layer_norm_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const long* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = decltype(tmp0)(tmp0 + 50265);
                    auto tmp2 = tmp0 < 0;
                    auto tmp3 = tmp2 ? tmp1 : tmp0;
                    TORCH_CHECK((0 <= tmp3) & (tmp3 < 50265L), "index out of bounds: 0 <= tmp3 < 50265L")
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*tmp3)));
                    auto tmp5 = static_cast<float>(1.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp7 + tmp8;
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp9);
                }
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                auto tmp10 = out_ptr0[static_cast<long>(x0)];
                auto tmp13 = out_ptr1[static_cast<long>(x0)];
                auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp1 = decltype(tmp0)(tmp0 + 50265);
                auto tmp2 = tmp0 < 0;
                auto tmp3 = tmp2 ? tmp1 : tmp0;
                TORCH_CHECK((0 <= tmp3) & (tmp3 < 50265L), "index out of bounds: 0 <= tmp3 < 50265L")
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*tmp3)));
                auto tmp5 = static_cast<float>(1.0);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp9 = tmp7 + tmp8;
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 - tmp11;
                auto tmp14 = static_cast<float>(512.0);
                auto tmp15 = tmp13 / tmp14;
                auto tmp16 = static_cast<float>(1e-05);
                auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                auto tmp18 = 1 / std::sqrt(tmp17);
                auto tmp19 = at::vec::Vectorized<float>(tmp18);
                auto tmp20 = tmp12 * tmp19;
                auto tmp22 = tmp20 * tmp21;
                auto tmp24 = tmp22 + tmp23;
                tmp24.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr1 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr2 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_native_layer_norm_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_gelu_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr1 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr2 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_native_layer_norm_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_gelu_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr1 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr2 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_native_layer_norm_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_gelu_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr1 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr2 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_native_layer_norm_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_gelu_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr1 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr2 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_native_layer_norm_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_gelu_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr1 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr2 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_native_layer_norm_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_gelu_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr1 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr2 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_native_layer_norm_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_gelu_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr1 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr2 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_native_layer_norm_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_gelu_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_arange_embedding_mul_native_layer_norm_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const long* in_ptr2,
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr2[static_cast<long>(x0)];
                    auto tmp1 = decltype(tmp0)(tmp0 + 50265);
                    auto tmp2 = tmp0 < 0;
                    auto tmp3 = tmp2 ? tmp1 : tmp0;
                    TORCH_CHECK((0 <= tmp3) & (tmp3 < 50265L), "index out of bounds: 0 <= tmp3 < 50265L")
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*tmp3)));
                    auto tmp5 = static_cast<float>(1.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp7);
                }
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = in_ptr2[static_cast<long>(x0)];
                auto tmp8 = out_ptr2[static_cast<long>(x0)];
                auto tmp11 = out_ptr3[static_cast<long>(x0)];
                auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = decltype(tmp0)(tmp0 + 50265);
                auto tmp2 = tmp0 < 0;
                auto tmp3 = tmp2 ? tmp1 : tmp0;
                TORCH_CHECK((0 <= tmp3) & (tmp3 < 50265L), "index out of bounds: 0 <= tmp3 < 50265L")
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*tmp3)));
                auto tmp5 = static_cast<float>(1.0);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 * tmp6;
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 - tmp9;
                auto tmp12 = static_cast<float>(512.0);
                auto tmp13 = tmp11 / tmp12;
                auto tmp14 = static_cast<float>(1e-05);
                auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                auto tmp16 = 1 / std::sqrt(tmp15);
                auto tmp17 = at::vec::Vectorized<float>(tmp16);
                auto tmp18 = tmp10 * tmp17;
                auto tmp20 = tmp18 * tmp19;
                auto tmp22 = tmp20 + tmp21;
                auto tmp24 = tmp22 + tmp23;
                tmp24.store(out_ptr4 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_41 = async_compile.cpp('''
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr1 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_42 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
}
''')


cpp_fused__softmax_clone_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                            tmp0.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (32L*x0) + (4096L*x1)));
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = in_ptr2[static_cast<long>(x0)];
                auto tmp6 = in_ptr3[static_cast<long>(x0)];
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 - tmp4;
                auto tmp7 = static_cast<float>(512.0);
                auto tmp8 = tmp6 / tmp7;
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr1 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr2 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_native_layer_norm_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_gelu_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_52 = async_compile.cpp('''
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr1 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
}
''')


cpp_fused__softmax_clone_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                            tmp0.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                        }
                    }
                }
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (32L*x0) + (4096L*x1)));
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr1 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr2 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_native_layer_norm_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_gelu_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_62 = async_compile.cpp('''
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr1 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
}
''')


cpp_fused__softmax_clone_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                            tmp0.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                        }
                    }
                }
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (32L*x0) + (4096L*x1)));
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr1 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr2 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_native_layer_norm_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_gelu_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_72 = async_compile.cpp('''
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr1 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
}
''')


cpp_fused__softmax_clone_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                            tmp0.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                        }
                    }
                }
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (32L*x0) + (4096L*x1)));
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr1 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr2 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_native_layer_norm_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_gelu_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_82 = async_compile.cpp('''
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr1 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
}
''')


cpp_fused__softmax_clone_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                            tmp0.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                        }
                    }
                }
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (32L*x0) + (4096L*x1)));
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_86 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_87 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr1 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr2 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_88 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_native_layer_norm_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_gelu_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_91 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_92 = async_compile.cpp('''
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr1 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
}
''')


cpp_fused__softmax_clone_94 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                            tmp0.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                        }
                    }
                }
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (32L*x0) + (4096L*x1)));
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_96 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_97 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr1 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr2 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_98 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_native_layer_norm_99 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_gelu_100 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_101 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_102 = async_compile.cpp('''
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr1 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
}
''')


cpp_fused__softmax_clone_104 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                            tmp0.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                        }
                    }
                }
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (32L*x0) + (4096L*x1)));
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_106 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_107 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr1 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr2 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_108 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_native_layer_norm_109 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_gelu_110 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_111 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_112 = async_compile.cpp('''
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr1 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                            auto tmp1 = c10::convert<long>(x2);
                            auto tmp2 = c10::convert<long>(1L + x1);
                            auto tmp3 = tmp1 < tmp2;
                            auto tmp4 = static_cast<float>(0.0);
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = tmp3 ? tmp4 : tmp5;
                            auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp7);
                        }
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = tmp_acc0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = c10::convert<long>(x2);
                        auto tmp2 = c10::convert<long>(1L + x1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<float>(0.0);
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        auto tmp7 = decltype(tmp0)(tmp0 + tmp6);
                        auto tmp9 = decltype(tmp7)(tmp7 - tmp8);
                        auto tmp10 = std::exp(tmp9);
                        in_out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp10;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
}
''')


cpp_fused__softmax_clone_114 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                            tmp0.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                        }
                    }
                }
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (32L*x0) + (4096L*x1)));
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_116 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_117 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr1 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr2 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_118 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_native_layer_norm_119 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_gelu_120 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_121 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                auto tmp11 = 1 / std::sqrt(tmp10);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp5 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused__log_softmax_add_nll_loss_forward_122 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const long* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       long* out_ptr3)
{
    auto out_ptr2 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(50264L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (50265L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(in_out_ptr0 + static_cast<long>(x1 + (50265L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(50264L); x1<static_cast<long>(50265L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (50265L*x0))];
                        auto tmp1 = in_ptr0[static_cast<long>(x1)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        in_out_ptr0[static_cast<long>(x1 + (50265L*x0))] = tmp2;
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(50264L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (50265L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(50264L); x1<static_cast<long>(50265L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (50265L*x0))];
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
                        auto tmp5 = decltype(tmp4)(tmp4 + 50265);
                        auto tmp6 = tmp4 < 0;
                        auto tmp7 = tmp6 ? tmp5 : tmp4;
                        TORCH_CHECK((0 <= tmp7) & (tmp7 < 50265L), "index out of bounds: 0 <= tmp7 < 50265L")
                        auto tmp8 = in_out_ptr0[static_cast<long>(tmp7 + (50265L*x0))];
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
                in_out_ptr1[static_cast<long>(0L)] = tmp3;
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1 = args
    args.clear()
    assert_size_stride(arg0_1, (512, 512), (512, 1))
    assert_size_stride(arg1_1, (512, 512), (512, 1))
    assert_size_stride(arg2_1, (50265, 512), (512, 1))
    assert_size_stride(arg3_1, (512, ), (1, ))
    assert_size_stride(arg4_1, (512, ), (1, ))
    assert_size_stride(arg5_1, (512, 512), (512, 1))
    assert_size_stride(arg6_1, (512, ), (1, ))
    assert_size_stride(arg7_1, (512, 512), (512, 1))
    assert_size_stride(arg8_1, (512, ), (1, ))
    assert_size_stride(arg9_1, (512, 512), (512, 1))
    assert_size_stride(arg10_1, (512, ), (1, ))
    assert_size_stride(arg11_1, (512, 512), (512, 1))
    assert_size_stride(arg12_1, (512, ), (1, ))
    assert_size_stride(arg13_1, (512, ), (1, ))
    assert_size_stride(arg14_1, (512, ), (1, ))
    assert_size_stride(arg15_1, (2048, 512), (512, 1))
    assert_size_stride(arg16_1, (2048, ), (1, ))
    assert_size_stride(arg17_1, (512, 2048), (2048, 1))
    assert_size_stride(arg18_1, (512, ), (1, ))
    assert_size_stride(arg19_1, (512, ), (1, ))
    assert_size_stride(arg20_1, (512, ), (1, ))
    assert_size_stride(arg21_1, (512, 512), (512, 1))
    assert_size_stride(arg22_1, (512, ), (1, ))
    assert_size_stride(arg23_1, (512, 512), (512, 1))
    assert_size_stride(arg24_1, (512, ), (1, ))
    assert_size_stride(arg25_1, (512, 512), (512, 1))
    assert_size_stride(arg26_1, (512, ), (1, ))
    assert_size_stride(arg27_1, (512, 512), (512, 1))
    assert_size_stride(arg28_1, (512, ), (1, ))
    assert_size_stride(arg29_1, (512, ), (1, ))
    assert_size_stride(arg30_1, (512, ), (1, ))
    assert_size_stride(arg31_1, (2048, 512), (512, 1))
    assert_size_stride(arg32_1, (2048, ), (1, ))
    assert_size_stride(arg33_1, (512, 2048), (2048, 1))
    assert_size_stride(arg34_1, (512, ), (1, ))
    assert_size_stride(arg35_1, (512, ), (1, ))
    assert_size_stride(arg36_1, (512, ), (1, ))
    assert_size_stride(arg37_1, (512, 512), (512, 1))
    assert_size_stride(arg38_1, (512, ), (1, ))
    assert_size_stride(arg39_1, (512, 512), (512, 1))
    assert_size_stride(arg40_1, (512, ), (1, ))
    assert_size_stride(arg41_1, (512, 512), (512, 1))
    assert_size_stride(arg42_1, (512, ), (1, ))
    assert_size_stride(arg43_1, (512, 512), (512, 1))
    assert_size_stride(arg44_1, (512, ), (1, ))
    assert_size_stride(arg45_1, (512, ), (1, ))
    assert_size_stride(arg46_1, (512, ), (1, ))
    assert_size_stride(arg47_1, (2048, 512), (512, 1))
    assert_size_stride(arg48_1, (2048, ), (1, ))
    assert_size_stride(arg49_1, (512, 2048), (2048, 1))
    assert_size_stride(arg50_1, (512, ), (1, ))
    assert_size_stride(arg51_1, (512, ), (1, ))
    assert_size_stride(arg52_1, (512, ), (1, ))
    assert_size_stride(arg53_1, (512, 512), (512, 1))
    assert_size_stride(arg54_1, (512, ), (1, ))
    assert_size_stride(arg55_1, (512, 512), (512, 1))
    assert_size_stride(arg56_1, (512, ), (1, ))
    assert_size_stride(arg57_1, (512, 512), (512, 1))
    assert_size_stride(arg58_1, (512, ), (1, ))
    assert_size_stride(arg59_1, (512, 512), (512, 1))
    assert_size_stride(arg60_1, (512, ), (1, ))
    assert_size_stride(arg61_1, (512, ), (1, ))
    assert_size_stride(arg62_1, (512, ), (1, ))
    assert_size_stride(arg63_1, (2048, 512), (512, 1))
    assert_size_stride(arg64_1, (2048, ), (1, ))
    assert_size_stride(arg65_1, (512, 2048), (2048, 1))
    assert_size_stride(arg66_1, (512, ), (1, ))
    assert_size_stride(arg67_1, (512, ), (1, ))
    assert_size_stride(arg68_1, (512, ), (1, ))
    assert_size_stride(arg69_1, (512, 512), (512, 1))
    assert_size_stride(arg70_1, (512, ), (1, ))
    assert_size_stride(arg71_1, (512, 512), (512, 1))
    assert_size_stride(arg72_1, (512, ), (1, ))
    assert_size_stride(arg73_1, (512, 512), (512, 1))
    assert_size_stride(arg74_1, (512, ), (1, ))
    assert_size_stride(arg75_1, (512, 512), (512, 1))
    assert_size_stride(arg76_1, (512, ), (1, ))
    assert_size_stride(arg77_1, (512, ), (1, ))
    assert_size_stride(arg78_1, (512, ), (1, ))
    assert_size_stride(arg79_1, (2048, 512), (512, 1))
    assert_size_stride(arg80_1, (2048, ), (1, ))
    assert_size_stride(arg81_1, (512, 2048), (2048, 1))
    assert_size_stride(arg82_1, (512, ), (1, ))
    assert_size_stride(arg83_1, (512, ), (1, ))
    assert_size_stride(arg84_1, (512, ), (1, ))
    assert_size_stride(arg85_1, (512, 512), (512, 1))
    assert_size_stride(arg86_1, (512, ), (1, ))
    assert_size_stride(arg87_1, (512, 512), (512, 1))
    assert_size_stride(arg88_1, (512, ), (1, ))
    assert_size_stride(arg89_1, (512, 512), (512, 1))
    assert_size_stride(arg90_1, (512, ), (1, ))
    assert_size_stride(arg91_1, (512, 512), (512, 1))
    assert_size_stride(arg92_1, (512, ), (1, ))
    assert_size_stride(arg93_1, (512, ), (1, ))
    assert_size_stride(arg94_1, (512, ), (1, ))
    assert_size_stride(arg95_1, (2048, 512), (512, 1))
    assert_size_stride(arg96_1, (2048, ), (1, ))
    assert_size_stride(arg97_1, (512, 2048), (2048, 1))
    assert_size_stride(arg98_1, (512, ), (1, ))
    assert_size_stride(arg99_1, (512, ), (1, ))
    assert_size_stride(arg100_1, (512, ), (1, ))
    assert_size_stride(arg101_1, (512, 512), (512, 1))
    assert_size_stride(arg102_1, (512, ), (1, ))
    assert_size_stride(arg103_1, (512, 512), (512, 1))
    assert_size_stride(arg104_1, (512, ), (1, ))
    assert_size_stride(arg105_1, (512, 512), (512, 1))
    assert_size_stride(arg106_1, (512, ), (1, ))
    assert_size_stride(arg107_1, (512, 512), (512, 1))
    assert_size_stride(arg108_1, (512, ), (1, ))
    assert_size_stride(arg109_1, (512, ), (1, ))
    assert_size_stride(arg110_1, (512, ), (1, ))
    assert_size_stride(arg111_1, (2048, 512), (512, 1))
    assert_size_stride(arg112_1, (2048, ), (1, ))
    assert_size_stride(arg113_1, (512, 2048), (2048, 1))
    assert_size_stride(arg114_1, (512, ), (1, ))
    assert_size_stride(arg115_1, (512, ), (1, ))
    assert_size_stride(arg116_1, (512, ), (1, ))
    assert_size_stride(arg117_1, (512, 512), (512, 1))
    assert_size_stride(arg118_1, (512, ), (1, ))
    assert_size_stride(arg119_1, (512, 512), (512, 1))
    assert_size_stride(arg120_1, (512, ), (1, ))
    assert_size_stride(arg121_1, (512, 512), (512, 1))
    assert_size_stride(arg122_1, (512, ), (1, ))
    assert_size_stride(arg123_1, (512, 512), (512, 1))
    assert_size_stride(arg124_1, (512, ), (1, ))
    assert_size_stride(arg125_1, (512, ), (1, ))
    assert_size_stride(arg126_1, (512, ), (1, ))
    assert_size_stride(arg127_1, (2048, 512), (512, 1))
    assert_size_stride(arg128_1, (2048, ), (1, ))
    assert_size_stride(arg129_1, (512, 2048), (2048, 1))
    assert_size_stride(arg130_1, (512, ), (1, ))
    assert_size_stride(arg131_1, (512, ), (1, ))
    assert_size_stride(arg132_1, (512, ), (1, ))
    assert_size_stride(arg133_1, (512, ), (1, ))
    assert_size_stride(arg134_1, (512, ), (1, ))
    assert_size_stride(arg135_1, (512, 512), (512, 1))
    assert_size_stride(arg136_1, (512, ), (1, ))
    assert_size_stride(arg137_1, (512, 512), (512, 1))
    assert_size_stride(arg138_1, (512, ), (1, ))
    assert_size_stride(arg139_1, (512, 512), (512, 1))
    assert_size_stride(arg140_1, (512, ), (1, ))
    assert_size_stride(arg141_1, (512, 512), (512, 1))
    assert_size_stride(arg142_1, (512, ), (1, ))
    assert_size_stride(arg143_1, (512, ), (1, ))
    assert_size_stride(arg144_1, (512, ), (1, ))
    assert_size_stride(arg145_1, (512, 512), (512, 1))
    assert_size_stride(arg146_1, (512, ), (1, ))
    assert_size_stride(arg147_1, (512, 512), (512, 1))
    assert_size_stride(arg148_1, (512, ), (1, ))
    assert_size_stride(arg149_1, (512, 512), (512, 1))
    assert_size_stride(arg150_1, (512, ), (1, ))
    assert_size_stride(arg151_1, (512, 512), (512, 1))
    assert_size_stride(arg152_1, (512, ), (1, ))
    assert_size_stride(arg153_1, (512, ), (1, ))
    assert_size_stride(arg154_1, (512, ), (1, ))
    assert_size_stride(arg155_1, (2048, 512), (512, 1))
    assert_size_stride(arg156_1, (2048, ), (1, ))
    assert_size_stride(arg157_1, (512, 2048), (2048, 1))
    assert_size_stride(arg158_1, (512, ), (1, ))
    assert_size_stride(arg159_1, (512, ), (1, ))
    assert_size_stride(arg160_1, (512, ), (1, ))
    assert_size_stride(arg161_1, (512, 512), (512, 1))
    assert_size_stride(arg162_1, (512, ), (1, ))
    assert_size_stride(arg163_1, (512, 512), (512, 1))
    assert_size_stride(arg164_1, (512, ), (1, ))
    assert_size_stride(arg165_1, (512, 512), (512, 1))
    assert_size_stride(arg166_1, (512, ), (1, ))
    assert_size_stride(arg167_1, (512, 512), (512, 1))
    assert_size_stride(arg168_1, (512, ), (1, ))
    assert_size_stride(arg169_1, (512, ), (1, ))
    assert_size_stride(arg170_1, (512, ), (1, ))
    assert_size_stride(arg171_1, (512, 512), (512, 1))
    assert_size_stride(arg172_1, (512, ), (1, ))
    assert_size_stride(arg173_1, (512, 512), (512, 1))
    assert_size_stride(arg174_1, (512, ), (1, ))
    assert_size_stride(arg175_1, (512, 512), (512, 1))
    assert_size_stride(arg176_1, (512, ), (1, ))
    assert_size_stride(arg177_1, (512, 512), (512, 1))
    assert_size_stride(arg178_1, (512, ), (1, ))
    assert_size_stride(arg179_1, (512, ), (1, ))
    assert_size_stride(arg180_1, (512, ), (1, ))
    assert_size_stride(arg181_1, (2048, 512), (512, 1))
    assert_size_stride(arg182_1, (2048, ), (1, ))
    assert_size_stride(arg183_1, (512, 2048), (2048, 1))
    assert_size_stride(arg184_1, (512, ), (1, ))
    assert_size_stride(arg185_1, (512, ), (1, ))
    assert_size_stride(arg186_1, (512, ), (1, ))
    assert_size_stride(arg187_1, (512, 512), (512, 1))
    assert_size_stride(arg188_1, (512, ), (1, ))
    assert_size_stride(arg189_1, (512, 512), (512, 1))
    assert_size_stride(arg190_1, (512, ), (1, ))
    assert_size_stride(arg191_1, (512, 512), (512, 1))
    assert_size_stride(arg192_1, (512, ), (1, ))
    assert_size_stride(arg193_1, (512, 512), (512, 1))
    assert_size_stride(arg194_1, (512, ), (1, ))
    assert_size_stride(arg195_1, (512, ), (1, ))
    assert_size_stride(arg196_1, (512, ), (1, ))
    assert_size_stride(arg197_1, (512, 512), (512, 1))
    assert_size_stride(arg198_1, (512, ), (1, ))
    assert_size_stride(arg199_1, (512, 512), (512, 1))
    assert_size_stride(arg200_1, (512, ), (1, ))
    assert_size_stride(arg201_1, (512, 512), (512, 1))
    assert_size_stride(arg202_1, (512, ), (1, ))
    assert_size_stride(arg203_1, (512, 512), (512, 1))
    assert_size_stride(arg204_1, (512, ), (1, ))
    assert_size_stride(arg205_1, (512, ), (1, ))
    assert_size_stride(arg206_1, (512, ), (1, ))
    assert_size_stride(arg207_1, (2048, 512), (512, 1))
    assert_size_stride(arg208_1, (2048, ), (1, ))
    assert_size_stride(arg209_1, (512, 2048), (2048, 1))
    assert_size_stride(arg210_1, (512, ), (1, ))
    assert_size_stride(arg211_1, (512, ), (1, ))
    assert_size_stride(arg212_1, (512, ), (1, ))
    assert_size_stride(arg213_1, (512, 512), (512, 1))
    assert_size_stride(arg214_1, (512, ), (1, ))
    assert_size_stride(arg215_1, (512, 512), (512, 1))
    assert_size_stride(arg216_1, (512, ), (1, ))
    assert_size_stride(arg217_1, (512, 512), (512, 1))
    assert_size_stride(arg218_1, (512, ), (1, ))
    assert_size_stride(arg219_1, (512, 512), (512, 1))
    assert_size_stride(arg220_1, (512, ), (1, ))
    assert_size_stride(arg221_1, (512, ), (1, ))
    assert_size_stride(arg222_1, (512, ), (1, ))
    assert_size_stride(arg223_1, (512, 512), (512, 1))
    assert_size_stride(arg224_1, (512, ), (1, ))
    assert_size_stride(arg225_1, (512, 512), (512, 1))
    assert_size_stride(arg226_1, (512, ), (1, ))
    assert_size_stride(arg227_1, (512, 512), (512, 1))
    assert_size_stride(arg228_1, (512, ), (1, ))
    assert_size_stride(arg229_1, (512, 512), (512, 1))
    assert_size_stride(arg230_1, (512, ), (1, ))
    assert_size_stride(arg231_1, (512, ), (1, ))
    assert_size_stride(arg232_1, (512, ), (1, ))
    assert_size_stride(arg233_1, (2048, 512), (512, 1))
    assert_size_stride(arg234_1, (2048, ), (1, ))
    assert_size_stride(arg235_1, (512, 2048), (2048, 1))
    assert_size_stride(arg236_1, (512, ), (1, ))
    assert_size_stride(arg237_1, (512, ), (1, ))
    assert_size_stride(arg238_1, (512, ), (1, ))
    assert_size_stride(arg239_1, (512, 512), (512, 1))
    assert_size_stride(arg240_1, (512, ), (1, ))
    assert_size_stride(arg241_1, (512, 512), (512, 1))
    assert_size_stride(arg242_1, (512, ), (1, ))
    assert_size_stride(arg243_1, (512, 512), (512, 1))
    assert_size_stride(arg244_1, (512, ), (1, ))
    assert_size_stride(arg245_1, (512, 512), (512, 1))
    assert_size_stride(arg246_1, (512, ), (1, ))
    assert_size_stride(arg247_1, (512, ), (1, ))
    assert_size_stride(arg248_1, (512, ), (1, ))
    assert_size_stride(arg249_1, (512, 512), (512, 1))
    assert_size_stride(arg250_1, (512, ), (1, ))
    assert_size_stride(arg251_1, (512, 512), (512, 1))
    assert_size_stride(arg252_1, (512, ), (1, ))
    assert_size_stride(arg253_1, (512, 512), (512, 1))
    assert_size_stride(arg254_1, (512, ), (1, ))
    assert_size_stride(arg255_1, (512, 512), (512, 1))
    assert_size_stride(arg256_1, (512, ), (1, ))
    assert_size_stride(arg257_1, (512, ), (1, ))
    assert_size_stride(arg258_1, (512, ), (1, ))
    assert_size_stride(arg259_1, (2048, 512), (512, 1))
    assert_size_stride(arg260_1, (2048, ), (1, ))
    assert_size_stride(arg261_1, (512, 2048), (2048, 1))
    assert_size_stride(arg262_1, (512, ), (1, ))
    assert_size_stride(arg263_1, (512, ), (1, ))
    assert_size_stride(arg264_1, (512, ), (1, ))
    assert_size_stride(arg265_1, (512, 512), (512, 1))
    assert_size_stride(arg266_1, (512, ), (1, ))
    assert_size_stride(arg267_1, (512, 512), (512, 1))
    assert_size_stride(arg268_1, (512, ), (1, ))
    assert_size_stride(arg269_1, (512, 512), (512, 1))
    assert_size_stride(arg270_1, (512, ), (1, ))
    assert_size_stride(arg271_1, (512, 512), (512, 1))
    assert_size_stride(arg272_1, (512, ), (1, ))
    assert_size_stride(arg273_1, (512, ), (1, ))
    assert_size_stride(arg274_1, (512, ), (1, ))
    assert_size_stride(arg275_1, (512, 512), (512, 1))
    assert_size_stride(arg276_1, (512, ), (1, ))
    assert_size_stride(arg277_1, (512, 512), (512, 1))
    assert_size_stride(arg278_1, (512, ), (1, ))
    assert_size_stride(arg279_1, (512, 512), (512, 1))
    assert_size_stride(arg280_1, (512, ), (1, ))
    assert_size_stride(arg281_1, (512, 512), (512, 1))
    assert_size_stride(arg282_1, (512, ), (1, ))
    assert_size_stride(arg283_1, (512, ), (1, ))
    assert_size_stride(arg284_1, (512, ), (1, ))
    assert_size_stride(arg285_1, (2048, 512), (512, 1))
    assert_size_stride(arg286_1, (2048, ), (1, ))
    assert_size_stride(arg287_1, (512, 2048), (2048, 1))
    assert_size_stride(arg288_1, (512, ), (1, ))
    assert_size_stride(arg289_1, (512, ), (1, ))
    assert_size_stride(arg290_1, (512, ), (1, ))
    assert_size_stride(arg291_1, (512, 512), (512, 1))
    assert_size_stride(arg292_1, (512, ), (1, ))
    assert_size_stride(arg293_1, (512, 512), (512, 1))
    assert_size_stride(arg294_1, (512, ), (1, ))
    assert_size_stride(arg295_1, (512, 512), (512, 1))
    assert_size_stride(arg296_1, (512, ), (1, ))
    assert_size_stride(arg297_1, (512, 512), (512, 1))
    assert_size_stride(arg298_1, (512, ), (1, ))
    assert_size_stride(arg299_1, (512, ), (1, ))
    assert_size_stride(arg300_1, (512, ), (1, ))
    assert_size_stride(arg301_1, (512, 512), (512, 1))
    assert_size_stride(arg302_1, (512, ), (1, ))
    assert_size_stride(arg303_1, (512, 512), (512, 1))
    assert_size_stride(arg304_1, (512, ), (1, ))
    assert_size_stride(arg305_1, (512, 512), (512, 1))
    assert_size_stride(arg306_1, (512, ), (1, ))
    assert_size_stride(arg307_1, (512, 512), (512, 1))
    assert_size_stride(arg308_1, (512, ), (1, ))
    assert_size_stride(arg309_1, (512, ), (1, ))
    assert_size_stride(arg310_1, (512, ), (1, ))
    assert_size_stride(arg311_1, (2048, 512), (512, 1))
    assert_size_stride(arg312_1, (2048, ), (1, ))
    assert_size_stride(arg313_1, (512, 2048), (2048, 1))
    assert_size_stride(arg314_1, (512, ), (1, ))
    assert_size_stride(arg315_1, (512, ), (1, ))
    assert_size_stride(arg316_1, (512, ), (1, ))
    assert_size_stride(arg317_1, (512, 512), (512, 1))
    assert_size_stride(arg318_1, (512, ), (1, ))
    assert_size_stride(arg319_1, (512, 512), (512, 1))
    assert_size_stride(arg320_1, (512, ), (1, ))
    assert_size_stride(arg321_1, (512, 512), (512, 1))
    assert_size_stride(arg322_1, (512, ), (1, ))
    assert_size_stride(arg323_1, (512, 512), (512, 1))
    assert_size_stride(arg324_1, (512, ), (1, ))
    assert_size_stride(arg325_1, (512, ), (1, ))
    assert_size_stride(arg326_1, (512, ), (1, ))
    assert_size_stride(arg327_1, (512, 512), (512, 1))
    assert_size_stride(arg328_1, (512, ), (1, ))
    assert_size_stride(arg329_1, (512, 512), (512, 1))
    assert_size_stride(arg330_1, (512, ), (1, ))
    assert_size_stride(arg331_1, (512, 512), (512, 1))
    assert_size_stride(arg332_1, (512, ), (1, ))
    assert_size_stride(arg333_1, (512, 512), (512, 1))
    assert_size_stride(arg334_1, (512, ), (1, ))
    assert_size_stride(arg335_1, (512, ), (1, ))
    assert_size_stride(arg336_1, (512, ), (1, ))
    assert_size_stride(arg337_1, (2048, 512), (512, 1))
    assert_size_stride(arg338_1, (2048, ), (1, ))
    assert_size_stride(arg339_1, (512, 2048), (2048, 1))
    assert_size_stride(arg340_1, (512, ), (1, ))
    assert_size_stride(arg341_1, (512, ), (1, ))
    assert_size_stride(arg342_1, (512, ), (1, ))
    assert_size_stride(arg343_1, (50265, 512), (512, 1))
    assert_size_stride(arg344_1, (1, 50265), (50265, 1))
    assert_size_stride(arg345_1, (1, 128), (128, 1))
    assert_size_stride(arg346_1, (1, 128), (128, 1))
    assert_size_stride(arg347_1, (1, 128), (128, 1))
    buf0 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf3 = empty((1, 128, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_arange_embedding_mul_native_layer_norm_0(c_void_p(arg347_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf3.data_ptr()))
    del arg0_1
    del arg347_1
    del arg3_1
    del arg4_1
    buf4 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_0_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg6_1, reinterpret_tensor(buf3, (128, 512), (512, 1), 0), reinterpret_tensor(arg5_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf4)
    del arg5_1
    del arg6_1
    buf5 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_0_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg8_1, reinterpret_tensor(buf3, (128, 512), (512, 1), 0), reinterpret_tensor(arg7_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf5)
    del arg7_1
    del arg8_1
    buf6 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_0_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg10_1, reinterpret_tensor(buf3, (128, 512), (512, 1), 0), reinterpret_tensor(arg9_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf6)
    del arg10_1
    del arg9_1
    buf7 = empty((1, 16, 128, 32), device='cpu', dtype=torch.float32)
    buf8 = empty((1, 16, 128, 32), device='cpu', dtype=torch.float32)
    buf9 = empty((1, 16, 128, 32), device='cpu', dtype=torch.float32)
    cpp_fused_clone_1(c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()))
    del buf4
    # Source Nodes: [], Original ATen: []
    buf10 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf7, (1, 16, 128, 32), (0, 4096, 32, 1), 0), reinterpret_tensor(buf8, (1, 16, 128, 32), (0, 4096, 32, 1), 0), reinterpret_tensor(buf9, (1, 16, 128, 32), (0, 4096, 32, 1), 0), scale=1.0)
    buf11 = buf10[0]
    del buf10
    buf18 = reinterpret_tensor(buf11, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf11  # reuse
    cpp_fused_clone_2(c_void_p(buf18.data_ptr()))
    buf19 = reinterpret_tensor(buf9, (128, 512), (512, 1), 0); del buf9  # reuse
    # Source Nodes: [hidden_states_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg12_1, reinterpret_tensor(buf18, (128, 512), (512, 1), 0), reinterpret_tensor(arg11_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf19)
    del arg11_1
    del arg12_1
    buf20 = buf1; del buf1  # reuse
    buf21 = buf0; del buf0  # reuse
    buf23 = reinterpret_tensor(buf18, (1, 128, 512), (65536, 512, 1), 0); del buf18  # reuse
    cpp_fused_add_native_layer_norm_3(c_void_p(buf3.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf23.data_ptr()))
    del arg13_1
    del arg14_1
    buf24 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___model_encoder_layers_0_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg16_1, reinterpret_tensor(buf23, (128, 512), (512, 1), 0), reinterpret_tensor(arg15_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf24)
    del arg15_1
    del arg16_1
    buf25 = reinterpret_tensor(buf24, (1, 128, 2048), (262144, 2048, 1), 0); del buf24  # reuse
    cpp_fused_gelu_4(c_void_p(buf25.data_ptr()))
    buf26 = reinterpret_tensor(buf3, (128, 512), (512, 1), 0); del buf3  # reuse
    # Source Nodes: [hidden_states_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg18_1, reinterpret_tensor(buf25, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg17_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf26)
    del arg17_1
    del arg18_1
    buf27 = buf21; del buf21  # reuse
    buf28 = buf20; del buf20  # reuse
    buf30 = reinterpret_tensor(buf19, (1, 128, 512), (65536, 512, 1), 0); del buf19  # reuse
    cpp_fused_add_native_layer_norm_5(c_void_p(buf23.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf30.data_ptr()))
    del arg19_1
    del arg20_1
    buf31 = buf26; del buf26  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_1_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg22_1, reinterpret_tensor(buf30, (128, 512), (512, 1), 0), reinterpret_tensor(arg21_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf31)
    del arg21_1
    del arg22_1
    buf32 = reinterpret_tensor(buf23, (128, 512), (512, 1), 0); del buf23  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_1_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg24_1, reinterpret_tensor(buf30, (128, 512), (512, 1), 0), reinterpret_tensor(arg23_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf32)
    del arg23_1
    del arg24_1
    buf33 = reinterpret_tensor(buf8, (128, 512), (512, 1), 0); del buf8  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_1_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg26_1, reinterpret_tensor(buf30, (128, 512), (512, 1), 0), reinterpret_tensor(arg25_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf33)
    del arg25_1
    del arg26_1
    buf34 = buf7; del buf7  # reuse
    buf35 = reinterpret_tensor(buf6, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf6  # reuse
    buf36 = reinterpret_tensor(buf5, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf5  # reuse
    cpp_fused_clone_6(c_void_p(buf31.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()))
    del buf31
    # Source Nodes: [], Original ATen: []
    buf37 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf34, (1, 16, 128, 32), (0, 4096, 32, 1), 0), reinterpret_tensor(buf35, (1, 16, 128, 32), (0, 4096, 32, 1), 0), reinterpret_tensor(buf36, (1, 16, 128, 32), (0, 4096, 32, 1), 0), scale=1.0)
    buf38 = buf37[0]
    del buf37
    buf45 = reinterpret_tensor(buf38, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf38  # reuse
    cpp_fused_clone_7(c_void_p(buf45.data_ptr()))
    buf46 = reinterpret_tensor(buf36, (128, 512), (512, 1), 0); del buf36  # reuse
    # Source Nodes: [hidden_states_14], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg28_1, reinterpret_tensor(buf45, (128, 512), (512, 1), 0), reinterpret_tensor(arg27_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf46)
    del arg27_1
    del arg28_1
    buf47 = buf28; del buf28  # reuse
    buf48 = buf27; del buf27  # reuse
    buf50 = reinterpret_tensor(buf45, (1, 128, 512), (65536, 512, 1), 0); del buf45  # reuse
    cpp_fused_add_native_layer_norm_8(c_void_p(buf30.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(arg29_1.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf50.data_ptr()))
    del arg29_1
    del arg30_1
    buf51 = reinterpret_tensor(buf25, (128, 2048), (2048, 1), 0); del buf25  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_1_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg32_1, reinterpret_tensor(buf50, (128, 512), (512, 1), 0), reinterpret_tensor(arg31_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf51)
    del arg31_1
    del arg32_1
    buf52 = reinterpret_tensor(buf51, (1, 128, 2048), (262144, 2048, 1), 0); del buf51  # reuse
    cpp_fused_gelu_9(c_void_p(buf52.data_ptr()))
    buf53 = buf46; del buf46  # reuse
    # Source Nodes: [hidden_states_20], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg34_1, reinterpret_tensor(buf52, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg33_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf53)
    del arg33_1
    del arg34_1
    buf54 = buf48; del buf48  # reuse
    buf55 = buf47; del buf47  # reuse
    buf57 = buf30; del buf30  # reuse
    cpp_fused_add_native_layer_norm_10(c_void_p(buf50.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(arg35_1.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf57.data_ptr()))
    del arg35_1
    del arg36_1
    buf58 = buf53; del buf53  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_2_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg38_1, reinterpret_tensor(buf57, (128, 512), (512, 1), 0), reinterpret_tensor(arg37_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf58)
    del arg37_1
    del arg38_1
    buf59 = reinterpret_tensor(buf50, (128, 512), (512, 1), 0); del buf50  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_2_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg40_1, reinterpret_tensor(buf57, (128, 512), (512, 1), 0), reinterpret_tensor(arg39_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf59)
    del arg39_1
    del arg40_1
    buf60 = reinterpret_tensor(buf35, (128, 512), (512, 1), 0); del buf35  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_2_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg42_1, reinterpret_tensor(buf57, (128, 512), (512, 1), 0), reinterpret_tensor(arg41_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf60)
    del arg41_1
    del arg42_1
    buf61 = buf34; del buf34  # reuse
    buf62 = reinterpret_tensor(buf33, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf33  # reuse
    buf63 = reinterpret_tensor(buf32, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf32  # reuse
    cpp_fused_clone_11(c_void_p(buf58.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf63.data_ptr()))
    del buf58
    # Source Nodes: [], Original ATen: []
    buf64 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf61, (1, 16, 128, 32), (0, 4096, 32, 1), 0), reinterpret_tensor(buf62, (1, 16, 128, 32), (0, 4096, 32, 1), 0), reinterpret_tensor(buf63, (1, 16, 128, 32), (0, 4096, 32, 1), 0), scale=1.0)
    buf65 = buf64[0]
    del buf64
    buf72 = reinterpret_tensor(buf65, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf65  # reuse
    cpp_fused_clone_12(c_void_p(buf72.data_ptr()))
    buf73 = reinterpret_tensor(buf63, (128, 512), (512, 1), 0); del buf63  # reuse
    # Source Nodes: [hidden_states_25], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg44_1, reinterpret_tensor(buf72, (128, 512), (512, 1), 0), reinterpret_tensor(arg43_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf73)
    del arg43_1
    del arg44_1
    buf74 = buf55; del buf55  # reuse
    buf75 = buf54; del buf54  # reuse
    buf77 = reinterpret_tensor(buf72, (1, 128, 512), (65536, 512, 1), 0); del buf72  # reuse
    cpp_fused_add_native_layer_norm_13(c_void_p(buf57.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(arg45_1.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf77.data_ptr()))
    del arg45_1
    del arg46_1
    buf78 = reinterpret_tensor(buf52, (128, 2048), (2048, 1), 0); del buf52  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_2_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg48_1, reinterpret_tensor(buf77, (128, 512), (512, 1), 0), reinterpret_tensor(arg47_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf78)
    del arg47_1
    del arg48_1
    buf79 = reinterpret_tensor(buf78, (1, 128, 2048), (262144, 2048, 1), 0); del buf78  # reuse
    cpp_fused_gelu_14(c_void_p(buf79.data_ptr()))
    buf80 = buf73; del buf73  # reuse
    # Source Nodes: [hidden_states_31], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg50_1, reinterpret_tensor(buf79, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg49_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf80)
    del arg49_1
    del arg50_1
    buf81 = buf75; del buf75  # reuse
    buf82 = buf74; del buf74  # reuse
    buf84 = buf57; del buf57  # reuse
    cpp_fused_add_native_layer_norm_15(c_void_p(buf77.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(arg51_1.data_ptr()), c_void_p(arg52_1.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf84.data_ptr()))
    del arg51_1
    del arg52_1
    buf85 = buf80; del buf80  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_3_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg54_1, reinterpret_tensor(buf84, (128, 512), (512, 1), 0), reinterpret_tensor(arg53_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf85)
    del arg53_1
    del arg54_1
    buf86 = reinterpret_tensor(buf77, (128, 512), (512, 1), 0); del buf77  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_3_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg56_1, reinterpret_tensor(buf84, (128, 512), (512, 1), 0), reinterpret_tensor(arg55_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf86)
    del arg55_1
    del arg56_1
    buf87 = reinterpret_tensor(buf62, (128, 512), (512, 1), 0); del buf62  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_3_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg58_1, reinterpret_tensor(buf84, (128, 512), (512, 1), 0), reinterpret_tensor(arg57_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf87)
    del arg57_1
    del arg58_1
    buf88 = buf61; del buf61  # reuse
    buf89 = reinterpret_tensor(buf60, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf60  # reuse
    buf90 = reinterpret_tensor(buf59, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf59  # reuse
    cpp_fused_clone_16(c_void_p(buf85.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf90.data_ptr()))
    del buf85
    # Source Nodes: [], Original ATen: []
    buf91 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf88, (1, 16, 128, 32), (0, 4096, 32, 1), 0), reinterpret_tensor(buf89, (1, 16, 128, 32), (0, 4096, 32, 1), 0), reinterpret_tensor(buf90, (1, 16, 128, 32), (0, 4096, 32, 1), 0), scale=1.0)
    buf92 = buf91[0]
    del buf91
    buf99 = reinterpret_tensor(buf92, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf92  # reuse
    cpp_fused_clone_17(c_void_p(buf99.data_ptr()))
    buf100 = reinterpret_tensor(buf90, (128, 512), (512, 1), 0); del buf90  # reuse
    # Source Nodes: [hidden_states_36], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg60_1, reinterpret_tensor(buf99, (128, 512), (512, 1), 0), reinterpret_tensor(arg59_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf100)
    del arg59_1
    del arg60_1
    buf101 = buf82; del buf82  # reuse
    buf102 = buf81; del buf81  # reuse
    buf104 = reinterpret_tensor(buf99, (1, 128, 512), (65536, 512, 1), 0); del buf99  # reuse
    cpp_fused_add_native_layer_norm_18(c_void_p(buf84.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(arg61_1.data_ptr()), c_void_p(arg62_1.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf104.data_ptr()))
    del arg61_1
    del arg62_1
    buf105 = reinterpret_tensor(buf79, (128, 2048), (2048, 1), 0); del buf79  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_3_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg64_1, reinterpret_tensor(buf104, (128, 512), (512, 1), 0), reinterpret_tensor(arg63_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf105)
    del arg63_1
    del arg64_1
    buf106 = reinterpret_tensor(buf105, (1, 128, 2048), (262144, 2048, 1), 0); del buf105  # reuse
    cpp_fused_gelu_19(c_void_p(buf106.data_ptr()))
    buf107 = reinterpret_tensor(buf84, (128, 512), (512, 1), 0); del buf84  # reuse
    # Source Nodes: [hidden_states_42], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg66_1, reinterpret_tensor(buf106, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg65_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf107)
    del arg65_1
    del arg66_1
    buf108 = buf102; del buf102  # reuse
    buf109 = buf101; del buf101  # reuse
    buf111 = reinterpret_tensor(buf100, (1, 128, 512), (65536, 512, 1), 0); del buf100  # reuse
    cpp_fused_add_native_layer_norm_20(c_void_p(buf104.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(arg67_1.data_ptr()), c_void_p(arg68_1.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf111.data_ptr()))
    del arg67_1
    del arg68_1
    buf112 = buf107; del buf107  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_4_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg70_1, reinterpret_tensor(buf111, (128, 512), (512, 1), 0), reinterpret_tensor(arg69_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf112)
    del arg69_1
    del arg70_1
    buf113 = reinterpret_tensor(buf104, (128, 512), (512, 1), 0); del buf104  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_4_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg72_1, reinterpret_tensor(buf111, (128, 512), (512, 1), 0), reinterpret_tensor(arg71_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf113)
    del arg71_1
    del arg72_1
    buf114 = reinterpret_tensor(buf89, (128, 512), (512, 1), 0); del buf89  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_4_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg74_1, reinterpret_tensor(buf111, (128, 512), (512, 1), 0), reinterpret_tensor(arg73_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf114)
    del arg73_1
    del arg74_1
    buf115 = buf88; del buf88  # reuse
    buf116 = reinterpret_tensor(buf87, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf87  # reuse
    buf117 = reinterpret_tensor(buf86, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf86  # reuse
    cpp_fused_clone_21(c_void_p(buf112.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()))
    del buf112
    # Source Nodes: [], Original ATen: []
    buf118 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf115, (1, 16, 128, 32), (0, 4096, 32, 1), 0), reinterpret_tensor(buf116, (1, 16, 128, 32), (0, 4096, 32, 1), 0), reinterpret_tensor(buf117, (1, 16, 128, 32), (0, 4096, 32, 1), 0), scale=1.0)
    buf119 = buf118[0]
    del buf118
    buf126 = reinterpret_tensor(buf119, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf119  # reuse
    cpp_fused_clone_22(c_void_p(buf126.data_ptr()))
    buf127 = reinterpret_tensor(buf117, (128, 512), (512, 1), 0); del buf117  # reuse
    # Source Nodes: [hidden_states_47], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg76_1, reinterpret_tensor(buf126, (128, 512), (512, 1), 0), reinterpret_tensor(arg75_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf127)
    del arg75_1
    del arg76_1
    buf128 = buf109; del buf109  # reuse
    buf129 = buf108; del buf108  # reuse
    buf131 = reinterpret_tensor(buf126, (1, 128, 512), (65536, 512, 1), 0); del buf126  # reuse
    cpp_fused_add_native_layer_norm_23(c_void_p(buf111.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(arg77_1.data_ptr()), c_void_p(arg78_1.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf131.data_ptr()))
    del arg77_1
    del arg78_1
    buf132 = reinterpret_tensor(buf106, (128, 2048), (2048, 1), 0); del buf106  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_4_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg80_1, reinterpret_tensor(buf131, (128, 512), (512, 1), 0), reinterpret_tensor(arg79_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf132)
    del arg79_1
    del arg80_1
    buf133 = reinterpret_tensor(buf132, (1, 128, 2048), (262144, 2048, 1), 0); del buf132  # reuse
    cpp_fused_gelu_24(c_void_p(buf133.data_ptr()))
    buf134 = buf127; del buf127  # reuse
    # Source Nodes: [hidden_states_53], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg82_1, reinterpret_tensor(buf133, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg81_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf134)
    del arg81_1
    del arg82_1
    buf135 = buf129; del buf129  # reuse
    buf136 = buf128; del buf128  # reuse
    buf138 = buf111; del buf111  # reuse
    cpp_fused_add_native_layer_norm_25(c_void_p(buf131.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(arg83_1.data_ptr()), c_void_p(arg84_1.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf138.data_ptr()))
    del arg83_1
    del arg84_1
    buf139 = buf134; del buf134  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_5_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg86_1, reinterpret_tensor(buf138, (128, 512), (512, 1), 0), reinterpret_tensor(arg85_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf139)
    del arg85_1
    del arg86_1
    buf140 = reinterpret_tensor(buf131, (128, 512), (512, 1), 0); del buf131  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_5_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg88_1, reinterpret_tensor(buf138, (128, 512), (512, 1), 0), reinterpret_tensor(arg87_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf140)
    del arg87_1
    del arg88_1
    buf141 = reinterpret_tensor(buf116, (128, 512), (512, 1), 0); del buf116  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_5_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg90_1, reinterpret_tensor(buf138, (128, 512), (512, 1), 0), reinterpret_tensor(arg89_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf141)
    del arg89_1
    del arg90_1
    buf142 = buf115; del buf115  # reuse
    buf143 = reinterpret_tensor(buf114, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf114  # reuse
    buf144 = reinterpret_tensor(buf113, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf113  # reuse
    cpp_fused_clone_26(c_void_p(buf139.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf144.data_ptr()))
    del buf139
    # Source Nodes: [], Original ATen: []
    buf145 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf142, (1, 16, 128, 32), (0, 4096, 32, 1), 0), reinterpret_tensor(buf143, (1, 16, 128, 32), (0, 4096, 32, 1), 0), reinterpret_tensor(buf144, (1, 16, 128, 32), (0, 4096, 32, 1), 0), scale=1.0)
    buf146 = buf145[0]
    del buf145
    buf153 = reinterpret_tensor(buf146, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf146  # reuse
    cpp_fused_clone_27(c_void_p(buf153.data_ptr()))
    buf154 = reinterpret_tensor(buf144, (128, 512), (512, 1), 0); del buf144  # reuse
    # Source Nodes: [hidden_states_58], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg92_1, reinterpret_tensor(buf153, (128, 512), (512, 1), 0), reinterpret_tensor(arg91_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf154)
    del arg91_1
    del arg92_1
    buf155 = buf136; del buf136  # reuse
    buf156 = buf135; del buf135  # reuse
    buf158 = reinterpret_tensor(buf153, (1, 128, 512), (65536, 512, 1), 0); del buf153  # reuse
    cpp_fused_add_native_layer_norm_28(c_void_p(buf138.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(arg93_1.data_ptr()), c_void_p(arg94_1.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf158.data_ptr()))
    del arg93_1
    del arg94_1
    buf159 = reinterpret_tensor(buf133, (128, 2048), (2048, 1), 0); del buf133  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_5_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg96_1, reinterpret_tensor(buf158, (128, 512), (512, 1), 0), reinterpret_tensor(arg95_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf159)
    del arg95_1
    del arg96_1
    buf160 = reinterpret_tensor(buf159, (1, 128, 2048), (262144, 2048, 1), 0); del buf159  # reuse
    cpp_fused_gelu_29(c_void_p(buf160.data_ptr()))
    buf161 = buf154; del buf154  # reuse
    # Source Nodes: [hidden_states_64], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg98_1, reinterpret_tensor(buf160, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg97_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf161)
    del arg97_1
    del arg98_1
    buf162 = buf156; del buf156  # reuse
    buf163 = buf155; del buf155  # reuse
    buf165 = buf138; del buf138  # reuse
    cpp_fused_add_native_layer_norm_30(c_void_p(buf158.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(arg99_1.data_ptr()), c_void_p(arg100_1.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf165.data_ptr()))
    del arg100_1
    del arg99_1
    buf166 = buf161; del buf161  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_6_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg102_1, reinterpret_tensor(buf165, (128, 512), (512, 1), 0), reinterpret_tensor(arg101_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf166)
    del arg101_1
    del arg102_1
    buf167 = reinterpret_tensor(buf158, (128, 512), (512, 1), 0); del buf158  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_6_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg104_1, reinterpret_tensor(buf165, (128, 512), (512, 1), 0), reinterpret_tensor(arg103_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf167)
    del arg103_1
    del arg104_1
    buf168 = reinterpret_tensor(buf143, (128, 512), (512, 1), 0); del buf143  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_6_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg106_1, reinterpret_tensor(buf165, (128, 512), (512, 1), 0), reinterpret_tensor(arg105_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf168)
    del arg105_1
    del arg106_1
    buf169 = buf142; del buf142  # reuse
    buf170 = reinterpret_tensor(buf141, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf141  # reuse
    buf171 = reinterpret_tensor(buf140, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf140  # reuse
    cpp_fused_clone_31(c_void_p(buf166.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf171.data_ptr()))
    del buf166
    # Source Nodes: [], Original ATen: []
    buf172 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf169, (1, 16, 128, 32), (0, 4096, 32, 1), 0), reinterpret_tensor(buf170, (1, 16, 128, 32), (0, 4096, 32, 1), 0), reinterpret_tensor(buf171, (1, 16, 128, 32), (0, 4096, 32, 1), 0), scale=1.0)
    buf173 = buf172[0]
    del buf172
    buf180 = reinterpret_tensor(buf173, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf173  # reuse
    cpp_fused_clone_32(c_void_p(buf180.data_ptr()))
    buf181 = reinterpret_tensor(buf171, (128, 512), (512, 1), 0); del buf171  # reuse
    # Source Nodes: [hidden_states_69], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg108_1, reinterpret_tensor(buf180, (128, 512), (512, 1), 0), reinterpret_tensor(arg107_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf181)
    del arg107_1
    del arg108_1
    buf182 = buf163; del buf163  # reuse
    buf183 = buf162; del buf162  # reuse
    buf185 = reinterpret_tensor(buf180, (1, 128, 512), (65536, 512, 1), 0); del buf180  # reuse
    cpp_fused_add_native_layer_norm_33(c_void_p(buf165.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(arg109_1.data_ptr()), c_void_p(arg110_1.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf185.data_ptr()))
    del arg109_1
    del arg110_1
    buf186 = reinterpret_tensor(buf160, (128, 2048), (2048, 1), 0); del buf160  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_6_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg112_1, reinterpret_tensor(buf185, (128, 512), (512, 1), 0), reinterpret_tensor(arg111_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf186)
    del arg111_1
    del arg112_1
    buf187 = reinterpret_tensor(buf186, (1, 128, 2048), (262144, 2048, 1), 0); del buf186  # reuse
    cpp_fused_gelu_34(c_void_p(buf187.data_ptr()))
    buf188 = buf181; del buf181  # reuse
    # Source Nodes: [hidden_states_75], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg114_1, reinterpret_tensor(buf187, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg113_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf188)
    del arg113_1
    del arg114_1
    buf189 = buf183; del buf183  # reuse
    buf190 = buf182; del buf182  # reuse
    buf192 = buf165; del buf165  # reuse
    cpp_fused_add_native_layer_norm_35(c_void_p(buf185.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(arg115_1.data_ptr()), c_void_p(arg116_1.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf192.data_ptr()))
    del arg115_1
    del arg116_1
    buf193 = buf188; del buf188  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_7_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg118_1, reinterpret_tensor(buf192, (128, 512), (512, 1), 0), reinterpret_tensor(arg117_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf193)
    del arg117_1
    del arg118_1
    buf194 = reinterpret_tensor(buf185, (128, 512), (512, 1), 0); del buf185  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_7_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg120_1, reinterpret_tensor(buf192, (128, 512), (512, 1), 0), reinterpret_tensor(arg119_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf194)
    del arg119_1
    del arg120_1
    buf195 = reinterpret_tensor(buf170, (128, 512), (512, 1), 0); del buf170  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_7_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg122_1, reinterpret_tensor(buf192, (128, 512), (512, 1), 0), reinterpret_tensor(arg121_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf195)
    del arg121_1
    del arg122_1
    buf196 = buf169; del buf169  # reuse
    buf197 = reinterpret_tensor(buf168, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf168  # reuse
    buf198 = reinterpret_tensor(buf167, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf167  # reuse
    cpp_fused_clone_36(c_void_p(buf193.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf198.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf199 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf196, (1, 16, 128, 32), (0, 4096, 32, 1), 0), reinterpret_tensor(buf197, (1, 16, 128, 32), (0, 4096, 32, 1), 0), reinterpret_tensor(buf198, (1, 16, 128, 32), (0, 4096, 32, 1), 0), scale=1.0)
    buf200 = buf199[0]
    del buf199
    buf207 = reinterpret_tensor(buf200, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf200  # reuse
    cpp_fused_clone_37(c_void_p(buf207.data_ptr()))
    buf208 = reinterpret_tensor(buf198, (128, 512), (512, 1), 0); del buf198  # reuse
    # Source Nodes: [hidden_states_80], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg124_1, reinterpret_tensor(buf207, (128, 512), (512, 1), 0), reinterpret_tensor(arg123_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf208)
    del arg123_1
    del arg124_1
    buf209 = buf190; del buf190  # reuse
    buf210 = buf189; del buf189  # reuse
    buf212 = reinterpret_tensor(buf207, (1, 128, 512), (65536, 512, 1), 0); del buf207  # reuse
    cpp_fused_add_native_layer_norm_38(c_void_p(buf192.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(arg125_1.data_ptr()), c_void_p(arg126_1.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf212.data_ptr()))
    del arg125_1
    del arg126_1
    buf213 = reinterpret_tensor(buf187, (128, 2048), (2048, 1), 0); del buf187  # reuse
    # Source Nodes: [l__mod___model_encoder_layers_7_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg128_1, reinterpret_tensor(buf212, (128, 512), (512, 1), 0), reinterpret_tensor(arg127_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf213)
    del arg127_1
    del arg128_1
    buf214 = reinterpret_tensor(buf213, (1, 128, 2048), (262144, 2048, 1), 0); del buf213  # reuse
    cpp_fused_gelu_39(c_void_p(buf214.data_ptr()))
    buf215 = buf208; del buf208  # reuse
    # Source Nodes: [hidden_states_86], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg130_1, reinterpret_tensor(buf214, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg129_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf215)
    del arg129_1
    del arg130_1
    buf216 = buf210; del buf210  # reuse
    buf217 = buf209; del buf209  # reuse
    buf219 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf220 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf222 = buf192; del buf192  # reuse
    cpp_fused_add_arange_embedding_mul_native_layer_norm_40(c_void_p(buf212.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(arg346_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg133_1.data_ptr()), c_void_p(arg134_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf222.data_ptr()))
    del arg133_1
    del arg134_1
    del arg1_1
    del arg2_1
    del arg346_1
    buf223 = reinterpret_tensor(buf197, (128, 512), (512, 1), 0); del buf197  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_0_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg136_1, reinterpret_tensor(buf222, (128, 512), (512, 1), 0), reinterpret_tensor(arg135_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf223)
    del arg135_1
    del arg136_1
    buf224 = reinterpret_tensor(buf196, (128, 512), (512, 1), 0); del buf196  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_0_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg138_1, reinterpret_tensor(buf222, (128, 512), (512, 1), 0), reinterpret_tensor(arg137_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf224)
    del arg137_1
    del arg138_1
    buf225 = reinterpret_tensor(buf195, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf195  # reuse
    buf226 = reinterpret_tensor(buf194, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf194  # reuse
    cpp_fused_clone_41(c_void_p(buf223.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf226.data_ptr()))
    buf227 = reinterpret_tensor(buf214, (16, 128, 128), (16384, 128, 1), 0); del buf214  # reuse
    # Source Nodes: [attn_weights_16], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf225, (16, 128, 32), (4096, 32, 1), 0), reinterpret_tensor(buf226, (16, 32, 128), (4096, 1, 32), 0), out=buf227)
    buf228 = empty_strided((16, 128, 1), (128, 1, 2048), device='cpu', dtype=torch.float32)
    buf229 = buf227; del buf227  # reuse
    buf230 = empty_strided((16, 128, 1), (128, 1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_42(c_void_p(buf229.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf230.data_ptr()))
    buf231 = reinterpret_tensor(buf226, (128, 512), (512, 1), 0); del buf226  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_0_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg140_1, reinterpret_tensor(buf222, (128, 512), (512, 1), 0), reinterpret_tensor(arg139_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf231)
    del arg139_1
    del arg140_1
    buf232 = buf229; del buf229  # reuse
    buf233 = buf225; del buf225  # reuse
    cpp_fused__softmax_clone_43(c_void_p(buf232.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf233.data_ptr()))
    buf234 = reinterpret_tensor(buf231, (16, 128, 32), (4096, 32, 1), 0); del buf231  # reuse
    # Source Nodes: [attn_output_40, attn_weights_19], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf232, reinterpret_tensor(buf233, (16, 128, 32), (4096, 32, 1), 0), out=buf234)
    buf235 = reinterpret_tensor(buf233, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf233  # reuse
    cpp_fused_clone_44(c_void_p(buf234.data_ptr()), c_void_p(buf235.data_ptr()))
    buf236 = reinterpret_tensor(buf234, (128, 512), (512, 1), 0); del buf234  # reuse
    # Source Nodes: [hidden_states_93], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg142_1, reinterpret_tensor(buf235, (128, 512), (512, 1), 0), reinterpret_tensor(arg141_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf236)
    del arg141_1
    del arg142_1
    buf237 = buf220; del buf220  # reuse
    buf238 = buf219; del buf219  # reuse
    buf240 = reinterpret_tensor(buf235, (1, 128, 512), (65536, 512, 1), 0); del buf235  # reuse
    cpp_fused_add_native_layer_norm_45(c_void_p(buf222.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(arg143_1.data_ptr()), c_void_p(arg144_1.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf240.data_ptr()))
    del arg143_1
    del arg144_1
    del buf237
    del buf238
    buf241 = buf236; del buf236  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_0_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg146_1, reinterpret_tensor(buf240, (128, 512), (512, 1), 0), reinterpret_tensor(arg145_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf241)
    del arg145_1
    del arg146_1
    buf242 = buf222; del buf222  # reuse
    cpp_fused_add_native_layer_norm_46(c_void_p(buf212.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(arg131_1.data_ptr()), c_void_p(arg132_1.data_ptr()), c_void_p(buf242.data_ptr()))
    del arg131_1
    del arg132_1
    buf243 = buf215; del buf215  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_0_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg148_1, reinterpret_tensor(buf242, (128, 512), (512, 1), 0), reinterpret_tensor(arg147_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf243)
    del arg147_1
    del arg148_1
    buf244 = reinterpret_tensor(buf212, (128, 512), (512, 1), 0); del buf212  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_0_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg150_1, reinterpret_tensor(buf242, (128, 512), (512, 1), 0), reinterpret_tensor(arg149_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf244)
    del arg149_1
    del arg150_1
    buf245 = reinterpret_tensor(buf224, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf224  # reuse
    buf246 = reinterpret_tensor(buf223, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf223  # reuse
    buf247 = reinterpret_tensor(buf193, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf193  # reuse
    cpp_fused_clone_47(c_void_p(buf241.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf247.data_ptr()))
    del buf241
    # Source Nodes: [], Original ATen: []
    buf248 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf245, (1, 16, 128, 32), (0, 4096, 32, 1), 0), reinterpret_tensor(buf246, (1, 16, 128, 32), (0, 4096, 32, 1), 0), reinterpret_tensor(buf247, (1, 16, 128, 32), (0, 4096, 32, 1), 0), scale=1.0)
    buf249 = buf248[0]
    del buf248
    buf256 = reinterpret_tensor(buf249, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf249  # reuse
    cpp_fused_clone_48(c_void_p(buf256.data_ptr()))
    buf257 = reinterpret_tensor(buf247, (128, 512), (512, 1), 0); del buf247  # reuse
    # Source Nodes: [hidden_states_97], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg152_1, reinterpret_tensor(buf256, (128, 512), (512, 1), 0), reinterpret_tensor(arg151_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf257)
    del arg151_1
    del arg152_1
    buf258 = buf217; del buf217  # reuse
    buf259 = buf216; del buf216  # reuse
    buf261 = reinterpret_tensor(buf256, (1, 128, 512), (65536, 512, 1), 0); del buf256  # reuse
    cpp_fused_add_native_layer_norm_49(c_void_p(buf240.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(arg153_1.data_ptr()), c_void_p(arg154_1.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf261.data_ptr()))
    del arg153_1
    del arg154_1
    buf262 = reinterpret_tensor(buf232, (128, 2048), (2048, 1), 0); del buf232  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_0_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg156_1, reinterpret_tensor(buf261, (128, 512), (512, 1), 0), reinterpret_tensor(arg155_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf262)
    del arg155_1
    del arg156_1
    buf263 = reinterpret_tensor(buf262, (1, 128, 2048), (262144, 2048, 1), 0); del buf262  # reuse
    cpp_fused_gelu_50(c_void_p(buf263.data_ptr()))
    buf264 = buf257; del buf257  # reuse
    # Source Nodes: [hidden_states_103], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg158_1, reinterpret_tensor(buf263, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg157_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf264)
    del arg157_1
    del arg158_1
    buf265 = buf259; del buf259  # reuse
    buf266 = buf258; del buf258  # reuse
    buf268 = buf240; del buf240  # reuse
    cpp_fused_add_native_layer_norm_51(c_void_p(buf261.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(arg159_1.data_ptr()), c_void_p(arg160_1.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf268.data_ptr()))
    del arg159_1
    del arg160_1
    buf269 = buf264; del buf264  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_1_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg162_1, reinterpret_tensor(buf268, (128, 512), (512, 1), 0), reinterpret_tensor(arg161_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf269)
    del arg161_1
    del arg162_1
    buf270 = reinterpret_tensor(buf261, (128, 512), (512, 1), 0); del buf261  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_1_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg164_1, reinterpret_tensor(buf268, (128, 512), (512, 1), 0), reinterpret_tensor(arg163_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf270)
    del arg163_1
    del arg164_1
    buf271 = buf246; del buf246  # reuse
    buf272 = buf245; del buf245  # reuse
    cpp_fused_clone_52(c_void_p(buf269.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf272.data_ptr()))
    buf273 = reinterpret_tensor(buf263, (16, 128, 128), (16384, 128, 1), 0); del buf263  # reuse
    # Source Nodes: [attn_weights_22], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf271, (16, 128, 32), (4096, 32, 1), 0), reinterpret_tensor(buf272, (16, 32, 128), (4096, 1, 32), 0), out=buf273)
    buf274 = buf230; del buf230  # reuse
    buf275 = buf273; del buf273  # reuse
    buf276 = buf228; del buf228  # reuse
    cpp_fused__softmax_53(c_void_p(buf275.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf276.data_ptr()))
    buf277 = reinterpret_tensor(buf272, (128, 512), (512, 1), 0); del buf272  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_1_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg166_1, reinterpret_tensor(buf268, (128, 512), (512, 1), 0), reinterpret_tensor(arg165_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf277)
    del arg165_1
    del arg166_1
    buf278 = buf275; del buf275  # reuse
    buf279 = buf271; del buf271  # reuse
    cpp_fused__softmax_clone_54(c_void_p(buf278.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf279.data_ptr()))
    buf280 = reinterpret_tensor(buf277, (16, 128, 32), (4096, 32, 1), 0); del buf277  # reuse
    # Source Nodes: [attn_output_50, attn_weights_25], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf278, reinterpret_tensor(buf279, (16, 128, 32), (4096, 32, 1), 0), out=buf280)
    buf281 = reinterpret_tensor(buf279, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf279  # reuse
    cpp_fused_clone_55(c_void_p(buf280.data_ptr()), c_void_p(buf281.data_ptr()))
    buf282 = reinterpret_tensor(buf280, (128, 512), (512, 1), 0); del buf280  # reuse
    # Source Nodes: [hidden_states_108], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg168_1, reinterpret_tensor(buf281, (128, 512), (512, 1), 0), reinterpret_tensor(arg167_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf282)
    del arg167_1
    del arg168_1
    buf283 = buf266; del buf266  # reuse
    buf284 = buf265; del buf265  # reuse
    buf286 = reinterpret_tensor(buf281, (1, 128, 512), (65536, 512, 1), 0); del buf281  # reuse
    cpp_fused_add_native_layer_norm_56(c_void_p(buf268.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(arg169_1.data_ptr()), c_void_p(arg170_1.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf286.data_ptr()))
    del arg169_1
    del arg170_1
    buf287 = buf282; del buf282  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_1_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg172_1, reinterpret_tensor(buf286, (128, 512), (512, 1), 0), reinterpret_tensor(arg171_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf287)
    del arg171_1
    del arg172_1
    buf288 = reinterpret_tensor(buf268, (128, 512), (512, 1), 0); del buf268  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_1_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg174_1, reinterpret_tensor(buf242, (128, 512), (512, 1), 0), reinterpret_tensor(arg173_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf288)
    del arg173_1
    del arg174_1
    buf289 = buf270; del buf270  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_1_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg176_1, reinterpret_tensor(buf242, (128, 512), (512, 1), 0), reinterpret_tensor(arg175_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf289)
    del arg175_1
    del arg176_1
    buf290 = reinterpret_tensor(buf269, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf269  # reuse
    buf291 = reinterpret_tensor(buf244, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf244  # reuse
    buf292 = reinterpret_tensor(buf243, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf243  # reuse
    cpp_fused_clone_57(c_void_p(buf287.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf292.data_ptr()))
    del buf287
    # Source Nodes: [], Original ATen: []
    buf293 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf290, (1, 16, 128, 32), (0, 4096, 32, 1), 0), reinterpret_tensor(buf291, (1, 16, 128, 32), (0, 4096, 32, 1), 0), reinterpret_tensor(buf292, (1, 16, 128, 32), (0, 4096, 32, 1), 0), scale=1.0)
    buf294 = buf293[0]
    del buf293
    buf301 = reinterpret_tensor(buf294, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf294  # reuse
    cpp_fused_clone_58(c_void_p(buf301.data_ptr()))
    buf302 = reinterpret_tensor(buf292, (128, 512), (512, 1), 0); del buf292  # reuse
    # Source Nodes: [hidden_states_112], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg178_1, reinterpret_tensor(buf301, (128, 512), (512, 1), 0), reinterpret_tensor(arg177_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf302)
    del arg177_1
    del arg178_1
    buf303 = buf284; del buf284  # reuse
    buf304 = buf283; del buf283  # reuse
    buf306 = reinterpret_tensor(buf301, (1, 128, 512), (65536, 512, 1), 0); del buf301  # reuse
    cpp_fused_add_native_layer_norm_59(c_void_p(buf286.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(arg179_1.data_ptr()), c_void_p(arg180_1.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf306.data_ptr()))
    del arg179_1
    del arg180_1
    buf307 = reinterpret_tensor(buf278, (128, 2048), (2048, 1), 0); del buf278  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_1_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg182_1, reinterpret_tensor(buf306, (128, 512), (512, 1), 0), reinterpret_tensor(arg181_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf307)
    del arg181_1
    del arg182_1
    buf308 = reinterpret_tensor(buf307, (1, 128, 2048), (262144, 2048, 1), 0); del buf307  # reuse
    cpp_fused_gelu_60(c_void_p(buf308.data_ptr()))
    buf309 = buf302; del buf302  # reuse
    # Source Nodes: [hidden_states_118], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg184_1, reinterpret_tensor(buf308, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg183_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf309)
    del arg183_1
    del arg184_1
    buf310 = buf304; del buf304  # reuse
    buf311 = buf303; del buf303  # reuse
    buf313 = buf286; del buf286  # reuse
    cpp_fused_add_native_layer_norm_61(c_void_p(buf306.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(arg185_1.data_ptr()), c_void_p(arg186_1.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf313.data_ptr()))
    del arg185_1
    del arg186_1
    buf314 = buf309; del buf309  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_2_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg188_1, reinterpret_tensor(buf313, (128, 512), (512, 1), 0), reinterpret_tensor(arg187_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf314)
    del arg187_1
    del arg188_1
    buf315 = reinterpret_tensor(buf306, (128, 512), (512, 1), 0); del buf306  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_2_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg190_1, reinterpret_tensor(buf313, (128, 512), (512, 1), 0), reinterpret_tensor(arg189_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf315)
    del arg189_1
    del arg190_1
    buf316 = buf291; del buf291  # reuse
    buf317 = buf290; del buf290  # reuse
    cpp_fused_clone_62(c_void_p(buf314.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf317.data_ptr()))
    buf318 = reinterpret_tensor(buf308, (16, 128, 128), (16384, 128, 1), 0); del buf308  # reuse
    # Source Nodes: [attn_weights_28], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf316, (16, 128, 32), (4096, 32, 1), 0), reinterpret_tensor(buf317, (16, 32, 128), (4096, 1, 32), 0), out=buf318)
    buf319 = buf276; del buf276  # reuse
    buf320 = buf318; del buf318  # reuse
    buf321 = buf274; del buf274  # reuse
    cpp_fused__softmax_63(c_void_p(buf320.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf321.data_ptr()))
    buf322 = reinterpret_tensor(buf317, (128, 512), (512, 1), 0); del buf317  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_2_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg192_1, reinterpret_tensor(buf313, (128, 512), (512, 1), 0), reinterpret_tensor(arg191_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf322)
    del arg191_1
    del arg192_1
    buf323 = buf320; del buf320  # reuse
    buf324 = buf316; del buf316  # reuse
    cpp_fused__softmax_clone_64(c_void_p(buf323.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf324.data_ptr()))
    buf325 = reinterpret_tensor(buf322, (16, 128, 32), (4096, 32, 1), 0); del buf322  # reuse
    # Source Nodes: [attn_output_60, attn_weights_31], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf323, reinterpret_tensor(buf324, (16, 128, 32), (4096, 32, 1), 0), out=buf325)
    buf326 = reinterpret_tensor(buf324, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf324  # reuse
    cpp_fused_clone_65(c_void_p(buf325.data_ptr()), c_void_p(buf326.data_ptr()))
    buf327 = reinterpret_tensor(buf325, (128, 512), (512, 1), 0); del buf325  # reuse
    # Source Nodes: [hidden_states_123], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg194_1, reinterpret_tensor(buf326, (128, 512), (512, 1), 0), reinterpret_tensor(arg193_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf327)
    del arg193_1
    del arg194_1
    buf328 = buf311; del buf311  # reuse
    buf329 = buf310; del buf310  # reuse
    buf331 = reinterpret_tensor(buf326, (1, 128, 512), (65536, 512, 1), 0); del buf326  # reuse
    cpp_fused_add_native_layer_norm_66(c_void_p(buf313.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(arg195_1.data_ptr()), c_void_p(arg196_1.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf331.data_ptr()))
    del arg195_1
    del arg196_1
    buf332 = buf327; del buf327  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_2_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg198_1, reinterpret_tensor(buf331, (128, 512), (512, 1), 0), reinterpret_tensor(arg197_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf332)
    del arg197_1
    del arg198_1
    buf333 = reinterpret_tensor(buf313, (128, 512), (512, 1), 0); del buf313  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_2_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg200_1, reinterpret_tensor(buf242, (128, 512), (512, 1), 0), reinterpret_tensor(arg199_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf333)
    del arg199_1
    del arg200_1
    buf334 = buf315; del buf315  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_2_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg202_1, reinterpret_tensor(buf242, (128, 512), (512, 1), 0), reinterpret_tensor(arg201_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf334)
    del arg201_1
    del arg202_1
    buf335 = reinterpret_tensor(buf314, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf314  # reuse
    buf336 = reinterpret_tensor(buf289, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf289  # reuse
    buf337 = reinterpret_tensor(buf288, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf288  # reuse
    cpp_fused_clone_67(c_void_p(buf332.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(buf337.data_ptr()))
    del buf332
    # Source Nodes: [], Original ATen: []
    buf338 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf335, (1, 16, 128, 32), (0, 4096, 32, 1), 0), reinterpret_tensor(buf336, (1, 16, 128, 32), (0, 4096, 32, 1), 0), reinterpret_tensor(buf337, (1, 16, 128, 32), (0, 4096, 32, 1), 0), scale=1.0)
    buf339 = buf338[0]
    del buf338
    buf346 = reinterpret_tensor(buf339, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf339  # reuse
    cpp_fused_clone_68(c_void_p(buf346.data_ptr()))
    buf347 = reinterpret_tensor(buf337, (128, 512), (512, 1), 0); del buf337  # reuse
    # Source Nodes: [hidden_states_127], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg204_1, reinterpret_tensor(buf346, (128, 512), (512, 1), 0), reinterpret_tensor(arg203_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf347)
    del arg203_1
    del arg204_1
    buf348 = buf329; del buf329  # reuse
    buf349 = buf328; del buf328  # reuse
    buf351 = reinterpret_tensor(buf346, (1, 128, 512), (65536, 512, 1), 0); del buf346  # reuse
    cpp_fused_add_native_layer_norm_69(c_void_p(buf331.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(arg205_1.data_ptr()), c_void_p(arg206_1.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf351.data_ptr()))
    del arg205_1
    del arg206_1
    buf352 = reinterpret_tensor(buf323, (128, 2048), (2048, 1), 0); del buf323  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_2_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg208_1, reinterpret_tensor(buf351, (128, 512), (512, 1), 0), reinterpret_tensor(arg207_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf352)
    del arg207_1
    del arg208_1
    buf353 = reinterpret_tensor(buf352, (1, 128, 2048), (262144, 2048, 1), 0); del buf352  # reuse
    cpp_fused_gelu_70(c_void_p(buf353.data_ptr()))
    buf354 = buf347; del buf347  # reuse
    # Source Nodes: [hidden_states_133], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg210_1, reinterpret_tensor(buf353, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg209_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf354)
    del arg209_1
    del arg210_1
    buf355 = buf349; del buf349  # reuse
    buf356 = buf348; del buf348  # reuse
    buf358 = buf331; del buf331  # reuse
    cpp_fused_add_native_layer_norm_71(c_void_p(buf351.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(arg211_1.data_ptr()), c_void_p(arg212_1.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf358.data_ptr()))
    del arg211_1
    del arg212_1
    buf359 = buf354; del buf354  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_3_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg214_1, reinterpret_tensor(buf358, (128, 512), (512, 1), 0), reinterpret_tensor(arg213_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf359)
    del arg213_1
    del arg214_1
    buf360 = reinterpret_tensor(buf351, (128, 512), (512, 1), 0); del buf351  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_3_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg216_1, reinterpret_tensor(buf358, (128, 512), (512, 1), 0), reinterpret_tensor(arg215_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf360)
    del arg215_1
    del arg216_1
    buf361 = buf336; del buf336  # reuse
    buf362 = buf335; del buf335  # reuse
    cpp_fused_clone_72(c_void_p(buf359.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf362.data_ptr()))
    buf363 = reinterpret_tensor(buf353, (16, 128, 128), (16384, 128, 1), 0); del buf353  # reuse
    # Source Nodes: [attn_weights_34], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf361, (16, 128, 32), (4096, 32, 1), 0), reinterpret_tensor(buf362, (16, 32, 128), (4096, 1, 32), 0), out=buf363)
    buf364 = buf321; del buf321  # reuse
    buf365 = buf363; del buf363  # reuse
    buf366 = buf319; del buf319  # reuse
    cpp_fused__softmax_73(c_void_p(buf365.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(buf366.data_ptr()))
    buf367 = reinterpret_tensor(buf362, (128, 512), (512, 1), 0); del buf362  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_3_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg218_1, reinterpret_tensor(buf358, (128, 512), (512, 1), 0), reinterpret_tensor(arg217_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf367)
    del arg217_1
    del arg218_1
    buf368 = buf365; del buf365  # reuse
    buf369 = buf361; del buf361  # reuse
    cpp_fused__softmax_clone_74(c_void_p(buf368.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf369.data_ptr()))
    buf370 = reinterpret_tensor(buf367, (16, 128, 32), (4096, 32, 1), 0); del buf367  # reuse
    # Source Nodes: [attn_output_70, attn_weights_37], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf368, reinterpret_tensor(buf369, (16, 128, 32), (4096, 32, 1), 0), out=buf370)
    buf371 = reinterpret_tensor(buf369, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf369  # reuse
    cpp_fused_clone_75(c_void_p(buf370.data_ptr()), c_void_p(buf371.data_ptr()))
    buf372 = reinterpret_tensor(buf370, (128, 512), (512, 1), 0); del buf370  # reuse
    # Source Nodes: [hidden_states_138], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg220_1, reinterpret_tensor(buf371, (128, 512), (512, 1), 0), reinterpret_tensor(arg219_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf372)
    del arg219_1
    del arg220_1
    buf373 = buf356; del buf356  # reuse
    buf374 = buf355; del buf355  # reuse
    buf376 = reinterpret_tensor(buf371, (1, 128, 512), (65536, 512, 1), 0); del buf371  # reuse
    cpp_fused_add_native_layer_norm_76(c_void_p(buf358.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(arg221_1.data_ptr()), c_void_p(arg222_1.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf376.data_ptr()))
    del arg221_1
    del arg222_1
    buf377 = buf372; del buf372  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_3_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg224_1, reinterpret_tensor(buf376, (128, 512), (512, 1), 0), reinterpret_tensor(arg223_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf377)
    del arg223_1
    del arg224_1
    buf378 = reinterpret_tensor(buf358, (128, 512), (512, 1), 0); del buf358  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_3_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg226_1, reinterpret_tensor(buf242, (128, 512), (512, 1), 0), reinterpret_tensor(arg225_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf378)
    del arg225_1
    del arg226_1
    buf379 = buf360; del buf360  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_3_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg228_1, reinterpret_tensor(buf242, (128, 512), (512, 1), 0), reinterpret_tensor(arg227_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf379)
    del arg227_1
    del arg228_1
    buf380 = reinterpret_tensor(buf359, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf359  # reuse
    buf381 = reinterpret_tensor(buf334, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf334  # reuse
    buf382 = reinterpret_tensor(buf333, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf333  # reuse
    cpp_fused_clone_77(c_void_p(buf377.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(buf382.data_ptr()))
    del buf377
    # Source Nodes: [], Original ATen: []
    buf383 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf380, (1, 16, 128, 32), (0, 4096, 32, 1), 0), reinterpret_tensor(buf381, (1, 16, 128, 32), (0, 4096, 32, 1), 0), reinterpret_tensor(buf382, (1, 16, 128, 32), (0, 4096, 32, 1), 0), scale=1.0)
    buf384 = buf383[0]
    del buf383
    buf391 = reinterpret_tensor(buf384, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf384  # reuse
    cpp_fused_clone_78(c_void_p(buf391.data_ptr()))
    buf392 = reinterpret_tensor(buf382, (128, 512), (512, 1), 0); del buf382  # reuse
    # Source Nodes: [hidden_states_142], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg230_1, reinterpret_tensor(buf391, (128, 512), (512, 1), 0), reinterpret_tensor(arg229_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf392)
    del arg229_1
    del arg230_1
    buf393 = buf374; del buf374  # reuse
    buf394 = buf373; del buf373  # reuse
    buf396 = reinterpret_tensor(buf391, (1, 128, 512), (65536, 512, 1), 0); del buf391  # reuse
    cpp_fused_add_native_layer_norm_79(c_void_p(buf376.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(arg231_1.data_ptr()), c_void_p(arg232_1.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(buf396.data_ptr()))
    del arg231_1
    del arg232_1
    buf397 = reinterpret_tensor(buf368, (128, 2048), (2048, 1), 0); del buf368  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_3_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg234_1, reinterpret_tensor(buf396, (128, 512), (512, 1), 0), reinterpret_tensor(arg233_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf397)
    del arg233_1
    del arg234_1
    buf398 = reinterpret_tensor(buf397, (1, 128, 2048), (262144, 2048, 1), 0); del buf397  # reuse
    cpp_fused_gelu_80(c_void_p(buf398.data_ptr()))
    buf399 = buf392; del buf392  # reuse
    # Source Nodes: [hidden_states_148], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg236_1, reinterpret_tensor(buf398, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg235_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf399)
    del arg235_1
    del arg236_1
    buf400 = buf394; del buf394  # reuse
    buf401 = buf393; del buf393  # reuse
    buf403 = buf376; del buf376  # reuse
    cpp_fused_add_native_layer_norm_81(c_void_p(buf396.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(arg237_1.data_ptr()), c_void_p(arg238_1.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(buf403.data_ptr()))
    del arg237_1
    del arg238_1
    buf404 = buf399; del buf399  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_4_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg240_1, reinterpret_tensor(buf403, (128, 512), (512, 1), 0), reinterpret_tensor(arg239_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf404)
    del arg239_1
    del arg240_1
    buf405 = reinterpret_tensor(buf396, (128, 512), (512, 1), 0); del buf396  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_4_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg242_1, reinterpret_tensor(buf403, (128, 512), (512, 1), 0), reinterpret_tensor(arg241_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf405)
    del arg241_1
    del arg242_1
    buf406 = buf381; del buf381  # reuse
    buf407 = buf380; del buf380  # reuse
    cpp_fused_clone_82(c_void_p(buf404.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(buf407.data_ptr()))
    buf408 = reinterpret_tensor(buf398, (16, 128, 128), (16384, 128, 1), 0); del buf398  # reuse
    # Source Nodes: [attn_weights_40], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf406, (16, 128, 32), (4096, 32, 1), 0), reinterpret_tensor(buf407, (16, 32, 128), (4096, 1, 32), 0), out=buf408)
    buf409 = buf366; del buf366  # reuse
    buf410 = buf408; del buf408  # reuse
    buf411 = buf364; del buf364  # reuse
    cpp_fused__softmax_83(c_void_p(buf410.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(buf411.data_ptr()))
    buf412 = reinterpret_tensor(buf407, (128, 512), (512, 1), 0); del buf407  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_4_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg244_1, reinterpret_tensor(buf403, (128, 512), (512, 1), 0), reinterpret_tensor(arg243_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf412)
    del arg243_1
    del arg244_1
    buf413 = buf410; del buf410  # reuse
    buf414 = buf406; del buf406  # reuse
    cpp_fused__softmax_clone_84(c_void_p(buf413.data_ptr()), c_void_p(buf411.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf414.data_ptr()))
    buf415 = reinterpret_tensor(buf412, (16, 128, 32), (4096, 32, 1), 0); del buf412  # reuse
    # Source Nodes: [attn_output_80, attn_weights_43], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf413, reinterpret_tensor(buf414, (16, 128, 32), (4096, 32, 1), 0), out=buf415)
    buf416 = reinterpret_tensor(buf414, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf414  # reuse
    cpp_fused_clone_85(c_void_p(buf415.data_ptr()), c_void_p(buf416.data_ptr()))
    buf417 = reinterpret_tensor(buf415, (128, 512), (512, 1), 0); del buf415  # reuse
    # Source Nodes: [hidden_states_153], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg246_1, reinterpret_tensor(buf416, (128, 512), (512, 1), 0), reinterpret_tensor(arg245_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf417)
    del arg245_1
    del arg246_1
    buf418 = buf401; del buf401  # reuse
    buf419 = buf400; del buf400  # reuse
    buf421 = reinterpret_tensor(buf416, (1, 128, 512), (65536, 512, 1), 0); del buf416  # reuse
    cpp_fused_add_native_layer_norm_86(c_void_p(buf403.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(arg247_1.data_ptr()), c_void_p(arg248_1.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf421.data_ptr()))
    del arg247_1
    del arg248_1
    buf422 = buf417; del buf417  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_4_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg250_1, reinterpret_tensor(buf421, (128, 512), (512, 1), 0), reinterpret_tensor(arg249_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf422)
    del arg249_1
    del arg250_1
    buf423 = reinterpret_tensor(buf403, (128, 512), (512, 1), 0); del buf403  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_4_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg252_1, reinterpret_tensor(buf242, (128, 512), (512, 1), 0), reinterpret_tensor(arg251_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf423)
    del arg251_1
    del arg252_1
    buf424 = buf405; del buf405  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_4_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg254_1, reinterpret_tensor(buf242, (128, 512), (512, 1), 0), reinterpret_tensor(arg253_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf424)
    del arg253_1
    del arg254_1
    buf425 = reinterpret_tensor(buf404, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf404  # reuse
    buf426 = reinterpret_tensor(buf379, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf379  # reuse
    buf427 = reinterpret_tensor(buf378, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf378  # reuse
    cpp_fused_clone_87(c_void_p(buf422.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf427.data_ptr()))
    del buf422
    # Source Nodes: [], Original ATen: []
    buf428 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf425, (1, 16, 128, 32), (0, 4096, 32, 1), 0), reinterpret_tensor(buf426, (1, 16, 128, 32), (0, 4096, 32, 1), 0), reinterpret_tensor(buf427, (1, 16, 128, 32), (0, 4096, 32, 1), 0), scale=1.0)
    buf429 = buf428[0]
    del buf428
    buf436 = reinterpret_tensor(buf429, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf429  # reuse
    cpp_fused_clone_88(c_void_p(buf436.data_ptr()))
    buf437 = reinterpret_tensor(buf427, (128, 512), (512, 1), 0); del buf427  # reuse
    # Source Nodes: [hidden_states_157], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg256_1, reinterpret_tensor(buf436, (128, 512), (512, 1), 0), reinterpret_tensor(arg255_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf437)
    del arg255_1
    del arg256_1
    buf438 = buf419; del buf419  # reuse
    buf439 = buf418; del buf418  # reuse
    buf441 = reinterpret_tensor(buf436, (1, 128, 512), (65536, 512, 1), 0); del buf436  # reuse
    cpp_fused_add_native_layer_norm_89(c_void_p(buf421.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(arg257_1.data_ptr()), c_void_p(arg258_1.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf441.data_ptr()))
    del arg257_1
    del arg258_1
    buf442 = reinterpret_tensor(buf413, (128, 2048), (2048, 1), 0); del buf413  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_4_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg260_1, reinterpret_tensor(buf441, (128, 512), (512, 1), 0), reinterpret_tensor(arg259_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf442)
    del arg259_1
    del arg260_1
    buf443 = reinterpret_tensor(buf442, (1, 128, 2048), (262144, 2048, 1), 0); del buf442  # reuse
    cpp_fused_gelu_90(c_void_p(buf443.data_ptr()))
    buf444 = buf437; del buf437  # reuse
    # Source Nodes: [hidden_states_163], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg262_1, reinterpret_tensor(buf443, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg261_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf444)
    del arg261_1
    del arg262_1
    buf445 = buf439; del buf439  # reuse
    buf446 = buf438; del buf438  # reuse
    buf448 = buf421; del buf421  # reuse
    cpp_fused_add_native_layer_norm_91(c_void_p(buf441.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(arg263_1.data_ptr()), c_void_p(arg264_1.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(buf446.data_ptr()), c_void_p(buf448.data_ptr()))
    del arg263_1
    del arg264_1
    buf449 = buf444; del buf444  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_5_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg266_1, reinterpret_tensor(buf448, (128, 512), (512, 1), 0), reinterpret_tensor(arg265_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf449)
    del arg265_1
    del arg266_1
    buf450 = reinterpret_tensor(buf441, (128, 512), (512, 1), 0); del buf441  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_5_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg268_1, reinterpret_tensor(buf448, (128, 512), (512, 1), 0), reinterpret_tensor(arg267_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf450)
    del arg267_1
    del arg268_1
    buf451 = buf426; del buf426  # reuse
    buf452 = buf425; del buf425  # reuse
    cpp_fused_clone_92(c_void_p(buf449.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(buf452.data_ptr()))
    buf453 = reinterpret_tensor(buf443, (16, 128, 128), (16384, 128, 1), 0); del buf443  # reuse
    # Source Nodes: [attn_weights_46], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf451, (16, 128, 32), (4096, 32, 1), 0), reinterpret_tensor(buf452, (16, 32, 128), (4096, 1, 32), 0), out=buf453)
    buf454 = buf411; del buf411  # reuse
    buf455 = buf453; del buf453  # reuse
    buf456 = buf409; del buf409  # reuse
    cpp_fused__softmax_93(c_void_p(buf455.data_ptr()), c_void_p(buf454.data_ptr()), c_void_p(buf456.data_ptr()))
    buf457 = reinterpret_tensor(buf452, (128, 512), (512, 1), 0); del buf452  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_5_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg270_1, reinterpret_tensor(buf448, (128, 512), (512, 1), 0), reinterpret_tensor(arg269_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf457)
    del arg269_1
    del arg270_1
    buf458 = buf455; del buf455  # reuse
    buf459 = buf451; del buf451  # reuse
    cpp_fused__softmax_clone_94(c_void_p(buf458.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(buf459.data_ptr()))
    buf460 = reinterpret_tensor(buf457, (16, 128, 32), (4096, 32, 1), 0); del buf457  # reuse
    # Source Nodes: [attn_output_90, attn_weights_49], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf458, reinterpret_tensor(buf459, (16, 128, 32), (4096, 32, 1), 0), out=buf460)
    buf461 = reinterpret_tensor(buf459, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf459  # reuse
    cpp_fused_clone_95(c_void_p(buf460.data_ptr()), c_void_p(buf461.data_ptr()))
    buf462 = reinterpret_tensor(buf460, (128, 512), (512, 1), 0); del buf460  # reuse
    # Source Nodes: [hidden_states_168], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg272_1, reinterpret_tensor(buf461, (128, 512), (512, 1), 0), reinterpret_tensor(arg271_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf462)
    del arg271_1
    del arg272_1
    buf463 = buf446; del buf446  # reuse
    buf464 = buf445; del buf445  # reuse
    buf466 = reinterpret_tensor(buf461, (1, 128, 512), (65536, 512, 1), 0); del buf461  # reuse
    cpp_fused_add_native_layer_norm_96(c_void_p(buf448.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(arg273_1.data_ptr()), c_void_p(arg274_1.data_ptr()), c_void_p(buf463.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(buf466.data_ptr()))
    del arg273_1
    del arg274_1
    buf467 = buf462; del buf462  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_5_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg276_1, reinterpret_tensor(buf466, (128, 512), (512, 1), 0), reinterpret_tensor(arg275_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf467)
    del arg275_1
    del arg276_1
    buf468 = reinterpret_tensor(buf448, (128, 512), (512, 1), 0); del buf448  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_5_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg278_1, reinterpret_tensor(buf242, (128, 512), (512, 1), 0), reinterpret_tensor(arg277_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf468)
    del arg277_1
    del arg278_1
    buf469 = buf450; del buf450  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_5_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg280_1, reinterpret_tensor(buf242, (128, 512), (512, 1), 0), reinterpret_tensor(arg279_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf469)
    del arg279_1
    del arg280_1
    buf470 = reinterpret_tensor(buf449, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf449  # reuse
    buf471 = reinterpret_tensor(buf424, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf424  # reuse
    buf472 = reinterpret_tensor(buf423, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf423  # reuse
    cpp_fused_clone_97(c_void_p(buf467.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf471.data_ptr()), c_void_p(buf472.data_ptr()))
    del buf467
    # Source Nodes: [], Original ATen: []
    buf473 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf470, (1, 16, 128, 32), (0, 4096, 32, 1), 0), reinterpret_tensor(buf471, (1, 16, 128, 32), (0, 4096, 32, 1), 0), reinterpret_tensor(buf472, (1, 16, 128, 32), (0, 4096, 32, 1), 0), scale=1.0)
    buf474 = buf473[0]
    del buf473
    buf481 = reinterpret_tensor(buf474, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf474  # reuse
    cpp_fused_clone_98(c_void_p(buf481.data_ptr()))
    buf482 = reinterpret_tensor(buf472, (128, 512), (512, 1), 0); del buf472  # reuse
    # Source Nodes: [hidden_states_172], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg282_1, reinterpret_tensor(buf481, (128, 512), (512, 1), 0), reinterpret_tensor(arg281_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf482)
    del arg281_1
    del arg282_1
    buf483 = buf464; del buf464  # reuse
    buf484 = buf463; del buf463  # reuse
    buf486 = reinterpret_tensor(buf481, (1, 128, 512), (65536, 512, 1), 0); del buf481  # reuse
    cpp_fused_add_native_layer_norm_99(c_void_p(buf466.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(arg283_1.data_ptr()), c_void_p(arg284_1.data_ptr()), c_void_p(buf483.data_ptr()), c_void_p(buf484.data_ptr()), c_void_p(buf486.data_ptr()))
    del arg283_1
    del arg284_1
    buf487 = reinterpret_tensor(buf458, (128, 2048), (2048, 1), 0); del buf458  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_5_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg286_1, reinterpret_tensor(buf486, (128, 512), (512, 1), 0), reinterpret_tensor(arg285_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf487)
    del arg285_1
    del arg286_1
    buf488 = reinterpret_tensor(buf487, (1, 128, 2048), (262144, 2048, 1), 0); del buf487  # reuse
    cpp_fused_gelu_100(c_void_p(buf488.data_ptr()))
    buf489 = buf482; del buf482  # reuse
    # Source Nodes: [hidden_states_178], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg288_1, reinterpret_tensor(buf488, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg287_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf489)
    del arg287_1
    del arg288_1
    buf490 = buf484; del buf484  # reuse
    buf491 = buf483; del buf483  # reuse
    buf493 = buf466; del buf466  # reuse
    cpp_fused_add_native_layer_norm_101(c_void_p(buf486.data_ptr()), c_void_p(buf489.data_ptr()), c_void_p(arg289_1.data_ptr()), c_void_p(arg290_1.data_ptr()), c_void_p(buf490.data_ptr()), c_void_p(buf491.data_ptr()), c_void_p(buf493.data_ptr()))
    del arg289_1
    del arg290_1
    buf494 = buf489; del buf489  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_6_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg292_1, reinterpret_tensor(buf493, (128, 512), (512, 1), 0), reinterpret_tensor(arg291_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf494)
    del arg291_1
    del arg292_1
    buf495 = reinterpret_tensor(buf486, (128, 512), (512, 1), 0); del buf486  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_6_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg294_1, reinterpret_tensor(buf493, (128, 512), (512, 1), 0), reinterpret_tensor(arg293_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf495)
    del arg293_1
    del arg294_1
    buf496 = buf471; del buf471  # reuse
    buf497 = buf470; del buf470  # reuse
    cpp_fused_clone_102(c_void_p(buf494.data_ptr()), c_void_p(buf495.data_ptr()), c_void_p(buf496.data_ptr()), c_void_p(buf497.data_ptr()))
    buf498 = reinterpret_tensor(buf488, (16, 128, 128), (16384, 128, 1), 0); del buf488  # reuse
    # Source Nodes: [attn_weights_52], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf496, (16, 128, 32), (4096, 32, 1), 0), reinterpret_tensor(buf497, (16, 32, 128), (4096, 1, 32), 0), out=buf498)
    buf499 = buf456; del buf456  # reuse
    buf500 = buf498; del buf498  # reuse
    buf501 = buf454; del buf454  # reuse
    cpp_fused__softmax_103(c_void_p(buf500.data_ptr()), c_void_p(buf499.data_ptr()), c_void_p(buf501.data_ptr()))
    buf502 = reinterpret_tensor(buf497, (128, 512), (512, 1), 0); del buf497  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_6_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg296_1, reinterpret_tensor(buf493, (128, 512), (512, 1), 0), reinterpret_tensor(arg295_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf502)
    del arg295_1
    del arg296_1
    buf503 = buf500; del buf500  # reuse
    buf504 = buf496; del buf496  # reuse
    cpp_fused__softmax_clone_104(c_void_p(buf503.data_ptr()), c_void_p(buf501.data_ptr()), c_void_p(buf502.data_ptr()), c_void_p(buf504.data_ptr()))
    buf505 = reinterpret_tensor(buf502, (16, 128, 32), (4096, 32, 1), 0); del buf502  # reuse
    # Source Nodes: [attn_output_100, attn_weights_55], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf503, reinterpret_tensor(buf504, (16, 128, 32), (4096, 32, 1), 0), out=buf505)
    buf506 = reinterpret_tensor(buf504, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf504  # reuse
    cpp_fused_clone_105(c_void_p(buf505.data_ptr()), c_void_p(buf506.data_ptr()))
    buf507 = reinterpret_tensor(buf505, (128, 512), (512, 1), 0); del buf505  # reuse
    # Source Nodes: [hidden_states_183], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg298_1, reinterpret_tensor(buf506, (128, 512), (512, 1), 0), reinterpret_tensor(arg297_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf507)
    del arg297_1
    del arg298_1
    buf508 = buf491; del buf491  # reuse
    buf509 = buf490; del buf490  # reuse
    buf511 = reinterpret_tensor(buf506, (1, 128, 512), (65536, 512, 1), 0); del buf506  # reuse
    cpp_fused_add_native_layer_norm_106(c_void_p(buf493.data_ptr()), c_void_p(buf507.data_ptr()), c_void_p(arg299_1.data_ptr()), c_void_p(arg300_1.data_ptr()), c_void_p(buf508.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(buf511.data_ptr()))
    del arg299_1
    del arg300_1
    buf512 = buf507; del buf507  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_6_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg302_1, reinterpret_tensor(buf511, (128, 512), (512, 1), 0), reinterpret_tensor(arg301_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf512)
    del arg301_1
    del arg302_1
    buf513 = reinterpret_tensor(buf493, (128, 512), (512, 1), 0); del buf493  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_6_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg304_1, reinterpret_tensor(buf242, (128, 512), (512, 1), 0), reinterpret_tensor(arg303_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf513)
    del arg303_1
    del arg304_1
    buf514 = buf495; del buf495  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_6_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg306_1, reinterpret_tensor(buf242, (128, 512), (512, 1), 0), reinterpret_tensor(arg305_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf514)
    del arg305_1
    del arg306_1
    buf515 = reinterpret_tensor(buf494, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf494  # reuse
    buf516 = reinterpret_tensor(buf469, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf469  # reuse
    buf517 = reinterpret_tensor(buf468, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf468  # reuse
    cpp_fused_clone_107(c_void_p(buf512.data_ptr()), c_void_p(buf513.data_ptr()), c_void_p(buf514.data_ptr()), c_void_p(buf515.data_ptr()), c_void_p(buf516.data_ptr()), c_void_p(buf517.data_ptr()))
    del buf512
    # Source Nodes: [], Original ATen: []
    buf518 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf515, (1, 16, 128, 32), (0, 4096, 32, 1), 0), reinterpret_tensor(buf516, (1, 16, 128, 32), (0, 4096, 32, 1), 0), reinterpret_tensor(buf517, (1, 16, 128, 32), (0, 4096, 32, 1), 0), scale=1.0)
    buf519 = buf518[0]
    del buf518
    buf526 = reinterpret_tensor(buf519, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf519  # reuse
    cpp_fused_clone_108(c_void_p(buf526.data_ptr()))
    buf527 = reinterpret_tensor(buf517, (128, 512), (512, 1), 0); del buf517  # reuse
    # Source Nodes: [hidden_states_187], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg308_1, reinterpret_tensor(buf526, (128, 512), (512, 1), 0), reinterpret_tensor(arg307_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf527)
    del arg307_1
    del arg308_1
    buf528 = buf509; del buf509  # reuse
    buf529 = buf508; del buf508  # reuse
    buf531 = reinterpret_tensor(buf526, (1, 128, 512), (65536, 512, 1), 0); del buf526  # reuse
    cpp_fused_add_native_layer_norm_109(c_void_p(buf511.data_ptr()), c_void_p(buf527.data_ptr()), c_void_p(arg309_1.data_ptr()), c_void_p(arg310_1.data_ptr()), c_void_p(buf528.data_ptr()), c_void_p(buf529.data_ptr()), c_void_p(buf531.data_ptr()))
    del arg309_1
    del arg310_1
    buf532 = reinterpret_tensor(buf503, (128, 2048), (2048, 1), 0); del buf503  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_6_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg312_1, reinterpret_tensor(buf531, (128, 512), (512, 1), 0), reinterpret_tensor(arg311_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf532)
    del arg311_1
    del arg312_1
    buf533 = reinterpret_tensor(buf532, (1, 128, 2048), (262144, 2048, 1), 0); del buf532  # reuse
    cpp_fused_gelu_110(c_void_p(buf533.data_ptr()))
    buf534 = buf527; del buf527  # reuse
    # Source Nodes: [hidden_states_193], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg314_1, reinterpret_tensor(buf533, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg313_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf534)
    del arg313_1
    del arg314_1
    buf535 = buf529; del buf529  # reuse
    buf536 = buf528; del buf528  # reuse
    buf538 = buf511; del buf511  # reuse
    cpp_fused_add_native_layer_norm_111(c_void_p(buf531.data_ptr()), c_void_p(buf534.data_ptr()), c_void_p(arg315_1.data_ptr()), c_void_p(arg316_1.data_ptr()), c_void_p(buf535.data_ptr()), c_void_p(buf536.data_ptr()), c_void_p(buf538.data_ptr()))
    del arg315_1
    del arg316_1
    buf539 = buf534; del buf534  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_7_self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg318_1, reinterpret_tensor(buf538, (128, 512), (512, 1), 0), reinterpret_tensor(arg317_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf539)
    del arg317_1
    del arg318_1
    buf540 = reinterpret_tensor(buf531, (128, 512), (512, 1), 0); del buf531  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_7_self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg320_1, reinterpret_tensor(buf538, (128, 512), (512, 1), 0), reinterpret_tensor(arg319_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf540)
    del arg319_1
    del arg320_1
    buf541 = buf516; del buf516  # reuse
    buf542 = buf515; del buf515  # reuse
    cpp_fused_clone_112(c_void_p(buf539.data_ptr()), c_void_p(buf540.data_ptr()), c_void_p(buf541.data_ptr()), c_void_p(buf542.data_ptr()))
    buf543 = reinterpret_tensor(buf533, (16, 128, 128), (16384, 128, 1), 0); del buf533  # reuse
    # Source Nodes: [attn_weights_58], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf541, (16, 128, 32), (4096, 32, 1), 0), reinterpret_tensor(buf542, (16, 32, 128), (4096, 1, 32), 0), out=buf543)
    buf544 = buf501; del buf501  # reuse
    buf545 = buf543; del buf543  # reuse
    buf546 = buf499; del buf499  # reuse
    cpp_fused__softmax_113(c_void_p(buf545.data_ptr()), c_void_p(buf544.data_ptr()), c_void_p(buf546.data_ptr()))
    del buf544
    buf547 = reinterpret_tensor(buf542, (128, 512), (512, 1), 0); del buf542  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_7_self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg322_1, reinterpret_tensor(buf538, (128, 512), (512, 1), 0), reinterpret_tensor(arg321_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf547)
    del arg321_1
    del arg322_1
    buf548 = buf545; del buf545  # reuse
    buf549 = buf541; del buf541  # reuse
    cpp_fused__softmax_clone_114(c_void_p(buf548.data_ptr()), c_void_p(buf546.data_ptr()), c_void_p(buf547.data_ptr()), c_void_p(buf549.data_ptr()))
    del buf546
    buf550 = reinterpret_tensor(buf547, (16, 128, 32), (4096, 32, 1), 0); del buf547  # reuse
    # Source Nodes: [attn_output_110, attn_weights_61], Original ATen: [aten._softmax, aten.bmm]
    extern_kernels.bmm(buf548, reinterpret_tensor(buf549, (16, 128, 32), (4096, 32, 1), 0), out=buf550)
    buf551 = reinterpret_tensor(buf549, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf549  # reuse
    cpp_fused_clone_115(c_void_p(buf550.data_ptr()), c_void_p(buf551.data_ptr()))
    buf552 = reinterpret_tensor(buf550, (128, 512), (512, 1), 0); del buf550  # reuse
    # Source Nodes: [hidden_states_198], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg324_1, reinterpret_tensor(buf551, (128, 512), (512, 1), 0), reinterpret_tensor(arg323_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf552)
    del arg323_1
    del arg324_1
    buf553 = buf536; del buf536  # reuse
    buf554 = buf535; del buf535  # reuse
    buf556 = reinterpret_tensor(buf551, (1, 128, 512), (65536, 512, 1), 0); del buf551  # reuse
    cpp_fused_add_native_layer_norm_116(c_void_p(buf538.data_ptr()), c_void_p(buf552.data_ptr()), c_void_p(arg325_1.data_ptr()), c_void_p(arg326_1.data_ptr()), c_void_p(buf553.data_ptr()), c_void_p(buf554.data_ptr()), c_void_p(buf556.data_ptr()))
    del arg325_1
    del arg326_1
    buf557 = buf552; del buf552  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_7_encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg328_1, reinterpret_tensor(buf556, (128, 512), (512, 1), 0), reinterpret_tensor(arg327_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf557)
    del arg327_1
    del arg328_1
    buf558 = reinterpret_tensor(buf538, (128, 512), (512, 1), 0); del buf538  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_7_encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg330_1, reinterpret_tensor(buf242, (128, 512), (512, 1), 0), reinterpret_tensor(arg329_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf558)
    del arg329_1
    del arg330_1
    buf559 = buf540; del buf540  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_7_encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg332_1, reinterpret_tensor(buf242, (128, 512), (512, 1), 0), reinterpret_tensor(arg331_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf559)
    del arg331_1
    del arg332_1
    buf560 = reinterpret_tensor(buf539, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf539  # reuse
    buf561 = reinterpret_tensor(buf514, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf514  # reuse
    buf562 = reinterpret_tensor(buf513, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf513  # reuse
    cpp_fused_clone_117(c_void_p(buf557.data_ptr()), c_void_p(buf558.data_ptr()), c_void_p(buf559.data_ptr()), c_void_p(buf560.data_ptr()), c_void_p(buf561.data_ptr()), c_void_p(buf562.data_ptr()))
    del buf557
    del buf558
    del buf559
    # Source Nodes: [], Original ATen: []
    buf563 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf560, (1, 16, 128, 32), (0, 4096, 32, 1), 0), reinterpret_tensor(buf561, (1, 16, 128, 32), (0, 4096, 32, 1), 0), reinterpret_tensor(buf562, (1, 16, 128, 32), (0, 4096, 32, 1), 0), scale=1.0)
    del buf560
    del buf561
    buf564 = buf563[0]
    del buf563
    buf571 = reinterpret_tensor(buf564, (1, 128, 16, 32), (65536, 512, 32, 1), 0); del buf564  # reuse
    cpp_fused_clone_118(c_void_p(buf571.data_ptr()))
    buf572 = reinterpret_tensor(buf562, (128, 512), (512, 1), 0); del buf562  # reuse
    # Source Nodes: [hidden_states_202], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg334_1, reinterpret_tensor(buf571, (128, 512), (512, 1), 0), reinterpret_tensor(arg333_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf572)
    del arg333_1
    del arg334_1
    buf573 = buf554; del buf554  # reuse
    buf574 = buf553; del buf553  # reuse
    buf576 = reinterpret_tensor(buf571, (1, 128, 512), (65536, 512, 1), 0); del buf571  # reuse
    cpp_fused_add_native_layer_norm_119(c_void_p(buf556.data_ptr()), c_void_p(buf572.data_ptr()), c_void_p(arg335_1.data_ptr()), c_void_p(arg336_1.data_ptr()), c_void_p(buf573.data_ptr()), c_void_p(buf574.data_ptr()), c_void_p(buf576.data_ptr()))
    del arg335_1
    del arg336_1
    buf577 = reinterpret_tensor(buf548, (128, 2048), (2048, 1), 0); del buf548  # reuse
    # Source Nodes: [l__mod___model_decoder_layers_7_fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg338_1, reinterpret_tensor(buf576, (128, 512), (512, 1), 0), reinterpret_tensor(arg337_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf577)
    del arg337_1
    del arg338_1
    buf578 = reinterpret_tensor(buf577, (1, 128, 2048), (262144, 2048, 1), 0); del buf577  # reuse
    cpp_fused_gelu_120(c_void_p(buf578.data_ptr()))
    buf579 = buf572; del buf572  # reuse
    # Source Nodes: [hidden_states_208], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg340_1, reinterpret_tensor(buf578, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg339_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf579)
    del arg339_1
    del arg340_1
    del buf578
    buf580 = buf574; del buf574  # reuse
    buf581 = buf573; del buf573  # reuse
    buf583 = buf556; del buf556  # reuse
    cpp_fused_add_native_layer_norm_121(c_void_p(buf576.data_ptr()), c_void_p(buf579.data_ptr()), c_void_p(arg341_1.data_ptr()), c_void_p(arg342_1.data_ptr()), c_void_p(buf580.data_ptr()), c_void_p(buf581.data_ptr()), c_void_p(buf583.data_ptr()))
    del arg341_1
    del arg342_1
    del buf576
    del buf579
    buf584 = empty((128, 50265), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___lm_head], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf583, (128, 512), (512, 1), 0), reinterpret_tensor(arg343_1, (512, 50265), (1, 512), 0), out=buf584)
    del arg343_1
    del buf583
    buf585 = reinterpret_tensor(buf584, (1, 128, 50265), (6433920, 50265, 1), 0); del buf584  # reuse
    buf586 = reinterpret_tensor(buf581, (128, 1), (1, 128), 0); del buf581  # reuse
    buf587 = reinterpret_tensor(buf580, (128, 1), (1, 128), 0); del buf580  # reuse
    buf588 = empty((), device='cpu', dtype=torch.float32)
    buf589 = empty((), device='cpu', dtype=torch.int64)
    buf590 = buf588; del buf588  # reuse
    cpp_fused__log_softmax_add_nll_loss_forward_122(c_void_p(buf585.data_ptr()), c_void_p(buf590.data_ptr()), c_void_p(arg344_1.data_ptr()), c_void_p(arg345_1.data_ptr()), c_void_p(buf586.data_ptr()), c_void_p(buf587.data_ptr()), c_void_p(buf589.data_ptr()))
    del arg344_1
    del arg345_1
    return (buf590, buf585, buf242, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((50265, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg263_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg266_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg267_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg269_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg270_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg271_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg272_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg273_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg274_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg275_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg276_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg277_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg278_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg279_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg280_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg281_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg282_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg283_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg284_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg285_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg286_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg287_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg288_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg289_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg290_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg291_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg292_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg293_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg294_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg295_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg296_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg297_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg298_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg299_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg300_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg301_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg302_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg303_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg304_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg305_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg306_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg307_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg308_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg309_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg310_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg311_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg312_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg313_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg314_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg315_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg316_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg317_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg318_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg319_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg320_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg321_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg322_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg323_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg324_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg325_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg326_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg327_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg328_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg329_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg330_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg331_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg332_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg333_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg334_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg335_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg336_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg337_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg338_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg339_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg340_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg341_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg342_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg343_1 = rand_strided((50265, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg344_1 = rand_strided((1, 50265), (50265, 1), device='cpu', dtype=torch.float32)
    arg345_1 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    arg346_1 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    arg347_1 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('BlenderbotSmallForConditionalGeneration', benchmark_compiled_module)
