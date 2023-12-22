
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


cpp_fused_cat_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const long* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = c10::convert<long>(x1);
                auto tmp1 = static_cast<long>(0);
                auto tmp2 = tmp0 >= tmp1;
                auto tmp3 = static_cast<long>(128);
                auto tmp4 = tmp0 < tmp3;
                auto tmp5 = [&]
                {
                    auto tmp6 = c10::convert<long>(x0);
                    auto tmp7 = static_cast<long>(127);
                    auto tmp8 = tmp6 < tmp7;
                    auto tmp9 = [&]
                    {
                        auto tmp10 = in_ptr0[static_cast<long>(1L + x0)];
                        auto tmp11 = decltype(tmp10)(tmp10 + 30522);
                        auto tmp12 = tmp10 < 0;
                        auto tmp13 = tmp12 ? tmp11 : tmp10;
                        TORCH_CHECK((0 <= tmp13) & (tmp13 < 30522L), "index out of bounds: 0 <= tmp13 < 30522L")
                        auto tmp14 = in_ptr1[static_cast<long>(x1 + (128L*tmp13))];
                        return tmp14;
                    }
                    ;
                    auto tmp15 = tmp8 ? tmp9() : static_cast<decltype(tmp9())>(0.0);
                    return tmp15;
                }
                ;
                auto tmp16 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                auto tmp17 = tmp0 >= tmp3;
                auto tmp18 = static_cast<long>(256);
                auto tmp19 = tmp0 < tmp18;
                auto tmp20 = tmp17 & tmp19;
                auto tmp21 = [&]
                {
                    auto tmp22 = in_ptr0[static_cast<long>(x0)];
                    auto tmp23 = decltype(tmp22)(tmp22 + 30522);
                    auto tmp24 = tmp22 < 0;
                    auto tmp25 = tmp24 ? tmp23 : tmp22;
                    TORCH_CHECK((0 <= tmp25) & (tmp25 < 30522L), "index out of bounds: 0 <= tmp25 < 30522L")
                    auto tmp26 = in_ptr1[static_cast<long>((-128L) + x1 + (128L*tmp25))];
                    return tmp26;
                }
                ;
                auto tmp27 = tmp20 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                auto tmp28 = tmp0 >= tmp18;
                auto tmp29 = static_cast<long>(384);
                auto tmp30 = tmp0 < tmp29;
                auto tmp31 = [&]
                {
                    auto tmp32 = c10::convert<long>((-1L) + x0);
                    auto tmp33 = static_cast<long>(0);
                    auto tmp34 = tmp32 >= tmp33;
                    auto tmp35 = [&]
                    {
                        auto tmp36 = in_ptr0[static_cast<long>((-1L) + x0)];
                        auto tmp37 = decltype(tmp36)(tmp36 + 30522);
                        auto tmp38 = tmp36 < 0;
                        auto tmp39 = tmp38 ? tmp37 : tmp36;
                        TORCH_CHECK((0 <= tmp39) & (tmp39 < 30522L), "index out of bounds: 0 <= tmp39 < 30522L")
                        auto tmp40 = in_ptr1[static_cast<long>((-256L) + x1 + (128L*tmp39))];
                        return tmp40;
                    }
                    ;
                    auto tmp41 = tmp34 ? tmp35() : static_cast<decltype(tmp35())>(0.0);
                    return tmp41;
                }
                ;
                auto tmp42 = tmp28 ? tmp31() : static_cast<decltype(tmp31())>(0.0);
                auto tmp43 = tmp20 ? tmp27 : tmp42;
                auto tmp44 = tmp4 ? tmp16 : tmp43;
                out_ptr0[static_cast<long>(x1 + (384L*x0))] = tmp44;
            }
        }
    }
}
''')


cpp_fused_add_embedding_mul_zeros_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = in_ptr0[static_cast<long>(x0)];
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = decltype(tmp1)(tmp1 + 512);
                auto tmp3 = tmp1 < 0;
                auto tmp4 = tmp3 ? tmp2 : tmp1;
                TORCH_CHECK((0 <= tmp4) & (tmp4 < 512L), "index out of bounds: 0 <= tmp4 < 512L")
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*tmp4)));
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                auto tmp10 = tmp8 * tmp9;
                auto tmp12 = tmp10 + tmp11;
                tmp12.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = tmp2 + tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            tmp0.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
            tmp0.store(in_out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 * tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = tmp6 * tmp7;
                auto tmp10 = tmp8 + tmp9;
                tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = tmp2 + tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            tmp0.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
            tmp0.store(in_out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 * tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = tmp6 * tmp7;
                auto tmp10 = tmp8 + tmp9;
                tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = tmp2 + tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            tmp0.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
            tmp0.store(in_out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 * tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = tmp6 * tmp7;
                auto tmp10 = tmp8 + tmp9;
                tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = tmp2 + tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            tmp0.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
            tmp0.store(in_out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 * tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = tmp6 * tmp7;
                auto tmp10 = tmp8 + tmp9;
                tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = tmp2 + tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            tmp0.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
            tmp0.store(in_out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 * tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = tmp6 * tmp7;
                auto tmp10 = tmp8 + tmp9;
                tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = tmp2 + tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            tmp0.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
            tmp0.store(in_out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 * tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = tmp6 * tmp7;
                auto tmp10 = tmp8 + tmp9;
                tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = tmp2 + tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            tmp0.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
            tmp0.store(in_out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 * tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = tmp6 * tmp7;
                auto tmp10 = tmp8 + tmp9;
                tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_82 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_83 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_85 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_86 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = tmp2 + tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_87 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            tmp0.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
            tmp0.store(in_out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_88 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 * tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = tmp6 * tmp7;
                auto tmp10 = tmp8 + tmp9;
                tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_91 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_93 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_94 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_95 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_96 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_97 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_98 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = tmp2 + tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_99 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            tmp0.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
            tmp0.store(in_out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_100 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 * tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = tmp6 * tmp7;
                auto tmp10 = tmp8 + tmp9;
                tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_101 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_102 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_103 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_104 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_105 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_106 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_107 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_108 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_109 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_110 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = tmp2 + tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_111 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            tmp0.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
            tmp0.store(in_out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_112 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 * tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = tmp6 * tmp7;
                auto tmp10 = tmp8 + tmp9;
                tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_113 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_114 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_115 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_116 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_117 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_118 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_119 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_120 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_121 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_122 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = tmp2 + tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_123 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            tmp0.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
            tmp0.store(in_out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_124 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 * tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = tmp6 * tmp7;
                auto tmp10 = tmp8 + tmp9;
                tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_125 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_126 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_127 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_128 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_129 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_130 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_131 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_132 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_133 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_134 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = tmp2 + tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_135 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            tmp0.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
            tmp0.store(in_out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_136 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 * tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = tmp6 * tmp7;
                auto tmp10 = tmp8 + tmp9;
                tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_137 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_138 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_139 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_140 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_141 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_142 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_143 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_144 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_145 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_146 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = tmp2 + tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_147 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            tmp0.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
            tmp0.store(in_out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_148 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 * tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = tmp6 * tmp7;
                auto tmp10 = tmp8 + tmp9;
                tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_149 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_150 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_151 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_152 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_153 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_154 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_155 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_156 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_157 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_158 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = tmp2 + tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_159 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            tmp0.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
            tmp0.store(in_out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_160 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 * tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = tmp6 * tmp7;
                auto tmp10 = tmp8 + tmp9;
                tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_161 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_162 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_163 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_164 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_165 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_166 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_167 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_168 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_169 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_170 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = tmp2 + tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_171 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            tmp0.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
            tmp0.store(in_out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_172 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 * tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = tmp6 * tmp7;
                auto tmp10 = tmp8 + tmp9;
                tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_173 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_174 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_175 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_176 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_177 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_178 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_179 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_180 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_181 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_182 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = tmp2 + tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_183 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            tmp0.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
            tmp0.store(in_out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_184 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 * tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = tmp6 * tmp7;
                auto tmp10 = tmp8 + tmp9;
                tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_185 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_186 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_187 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_188 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_189 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_190 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_191 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_192 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_193 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_194 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = tmp2 + tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_195 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            tmp0.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
            tmp0.store(in_out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_196 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 * tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = tmp6 * tmp7;
                auto tmp10 = tmp8 + tmp9;
                tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_197 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_198 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_199 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_200 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_201 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_202 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_203 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_204 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_205 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_206 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = tmp2 + tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_207 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            tmp0.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
            tmp0.store(in_out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_208 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 * tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = tmp6 * tmp7;
                auto tmp10 = tmp8 + tmp9;
                tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_209 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_210 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_211 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_212 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_213 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_214 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_215 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_216 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_217 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_218 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = tmp2 + tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_219 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            tmp0.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
            tmp0.store(in_out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_220 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 * tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = tmp6 * tmp7;
                auto tmp10 = tmp8 + tmp9;
                tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_221 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_222 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_223 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_224 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_225 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_226 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_227 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_228 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_229 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_230 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = tmp2 + tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_231 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            tmp0.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
            tmp0.store(in_out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_232 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 * tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = tmp6 * tmp7;
                auto tmp10 = tmp8 + tmp9;
                tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_233 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_234 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_235 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_236 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_237 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_238 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_239 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_240 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_241 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_242 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = tmp2 + tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_243 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            tmp0.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
            tmp0.store(in_out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_244 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 * tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = tmp6 * tmp7;
                auto tmp10 = tmp8 + tmp9;
                tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_245 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_246 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_247 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_248 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_249 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_250 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_251 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_252 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_253 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_254 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = tmp2 + tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_255 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            tmp0.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
            tmp0.store(in_out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_256 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 * tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = tmp6 * tmp7;
                auto tmp10 = tmp8 + tmp9;
                tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_257 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_258 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_259 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_260 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_261 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_262 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_263 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_264 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_265 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_266 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = tmp2 + tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_267 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            tmp0.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
            tmp0.store(in_out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_268 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 * tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = tmp6 * tmp7;
                auto tmp10 = tmp8 + tmp9;
                tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_269 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_270 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_271 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_272 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_273 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_274 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_275 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_276 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_277 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_278 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = tmp2 + tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_279 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            tmp0.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            tmp0.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
            tmp0.store(in_out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_280 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 * tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = tmp6 * tmp7;
                auto tmp10 = tmp8 + tmp9;
                tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_281 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_282 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_283 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_284 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_285 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_286 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_relu_287 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(65536L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_mul_288 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_mul_289 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 * tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused__log_softmax_add_clamp_clone_div_nll_loss_forward_290 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const long* in_ptr1,
                       const long* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    auto out_ptr1 = in_out_ptr0;
    {
        {
            #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
            float tmp_acc0 = -std::numeric_limits<float>::infinity();
            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((2L*x0) + (2L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
            }
            tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
            out_ptr1[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
        }
    }
    {
        {
            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
            float tmp_acc0 = 0;
            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                auto tmp1 = out_ptr1[static_cast<long>(0L)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 - tmp2;
                auto tmp4 = tmp3.exp();
                tmp_acc0_vec = tmp_acc0_vec + tmp4;
            }
            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
            out_ptr2[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
        }
    }
    {
        {
            #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
            float tmp_acc0 = -std::numeric_limits<float>::infinity();
            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(1L + (2L*x0) + (2L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr3 + static_cast<long>(x0));
                tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
            }
            tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
            out_ptr4[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
        }
    }
    {
        {
            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
            float tmp_acc0 = 0;
            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = out_ptr4[static_cast<long>(0L)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 - tmp2;
                auto tmp4 = tmp3.exp();
                tmp_acc0_vec = tmp_acc0_vec + tmp4;
            }
            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
            out_ptr5[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
        }
    }
    {
        auto tmp0 = in_ptr1[static_cast<long>(0L)];
        auto tmp11 = out_ptr1[static_cast<long>(0L)];
        auto tmp13 = out_ptr2[static_cast<long>(0L)];
        auto tmp22 = in_ptr2[static_cast<long>(0L)];
        auto tmp31 = out_ptr4[static_cast<long>(0L)];
        auto tmp33 = out_ptr5[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(0);
        auto tmp2 = max_propagate_nan(tmp0, tmp1);
        auto tmp3 = static_cast<long>(128);
        auto tmp4 = min_propagate_nan(tmp2, tmp3);
        auto tmp5 = tmp4 != tmp3;
        auto tmp6 = tmp5 ? tmp4 : tmp1;
        auto tmp7 = decltype(tmp6)(tmp6 + 128);
        auto tmp8 = tmp6 < 0;
        auto tmp9 = tmp8 ? tmp7 : tmp6;
        TORCH_CHECK((0 <= tmp9) & (tmp9 < 128L), "index out of bounds: 0 <= tmp9 < 128L")
        auto tmp10 = out_ptr0[static_cast<long>(tmp9)];
        auto tmp12 = decltype(tmp10)(tmp10 - tmp11);
        auto tmp14 = std::log(tmp13);
        auto tmp15 = decltype(tmp12)(tmp12 - tmp14);
        auto tmp16 = decltype(tmp15)(-tmp15);
        auto tmp17 = static_cast<float>(0.0);
        auto tmp18 = tmp5 ? tmp16 : tmp17;
        auto tmp19 = c10::convert<long>(tmp5);
        auto tmp20 = c10::convert<float>(tmp19);
        auto tmp21 = tmp18 / tmp20;
        auto tmp23 = max_propagate_nan(tmp22, tmp1);
        auto tmp24 = min_propagate_nan(tmp23, tmp3);
        auto tmp25 = tmp24 != tmp3;
        auto tmp26 = tmp25 ? tmp24 : tmp1;
        auto tmp27 = decltype(tmp26)(tmp26 + 128);
        auto tmp28 = tmp26 < 0;
        auto tmp29 = tmp28 ? tmp27 : tmp26;
        TORCH_CHECK((0 <= tmp29) & (tmp29 < 128L), "index out of bounds: 0 <= tmp29 < 128L")
        auto tmp30 = out_ptr3[static_cast<long>(tmp29)];
        auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
        auto tmp34 = std::log(tmp33);
        auto tmp35 = decltype(tmp32)(tmp32 - tmp34);
        auto tmp36 = decltype(tmp35)(-tmp35);
        auto tmp37 = tmp25 ? tmp36 : tmp17;
        auto tmp38 = c10::convert<long>(tmp25);
        auto tmp39 = c10::convert<float>(tmp38);
        auto tmp40 = tmp37 / tmp39;
        auto tmp41 = decltype(tmp21)(tmp21 + tmp40);
        auto tmp42 = static_cast<float>(2.0);
        auto tmp43 = tmp41 / tmp42;
        in_out_ptr0[static_cast<long>(0L)] = tmp43;
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1, arg797_1, arg798_1, arg799_1, arg800_1, arg801_1, arg802_1, arg803_1, arg804_1, arg805_1, arg806_1, arg807_1, arg808_1, arg809_1, arg810_1, arg811_1, arg812_1, arg813_1, arg814_1, arg815_1, arg816_1, arg817_1, arg818_1, arg819_1, arg820_1, arg821_1, arg822_1, arg823_1, arg824_1, arg825_1, arg826_1, arg827_1, arg828_1, arg829_1, arg830_1, arg831_1, arg832_1, arg833_1, arg834_1, arg835_1, arg836_1, arg837_1, arg838_1, arg839_1, arg840_1, arg841_1, arg842_1, arg843_1, arg844_1, arg845_1, arg846_1, arg847_1, arg848_1, arg849_1, arg850_1, arg851_1, arg852_1, arg853_1, arg854_1, arg855_1, arg856_1, arg857_1, arg858_1, arg859_1, arg860_1, arg861_1, arg862_1, arg863_1, arg864_1, arg865_1, arg866_1, arg867_1, arg868_1, arg869_1, arg870_1, arg871_1, arg872_1, arg873_1, arg874_1, arg875_1, arg876_1, arg877_1, arg878_1, arg879_1, arg880_1, arg881_1, arg882_1, arg883_1, arg884_1, arg885_1, arg886_1, arg887_1, arg888_1, arg889_1, arg890_1, arg891_1, arg892_1, arg893_1, arg894_1, arg895_1, arg896_1, arg897_1, arg898_1, arg899_1, arg900_1, arg901_1, arg902_1, arg903_1, arg904_1, arg905_1, arg906_1, arg907_1, arg908_1, arg909_1, arg910_1, arg911_1, arg912_1, arg913_1, arg914_1, arg915_1, arg916_1, arg917_1, arg918_1, arg919_1, arg920_1, arg921_1, arg922_1, arg923_1, arg924_1, arg925_1, arg926_1, arg927_1, arg928_1, arg929_1, arg930_1, arg931_1, arg932_1, arg933_1, arg934_1, arg935_1, arg936_1, arg937_1, arg938_1, arg939_1, arg940_1, arg941_1, arg942_1, arg943_1, arg944_1, arg945_1, arg946_1, arg947_1, arg948_1, arg949_1, arg950_1, arg951_1, arg952_1, arg953_1, arg954_1, arg955_1, arg956_1, arg957_1, arg958_1, arg959_1, arg960_1, arg961_1, arg962_1, arg963_1, arg964_1, arg965_1, arg966_1, arg967_1, arg968_1, arg969_1, arg970_1, arg971_1, arg972_1, arg973_1, arg974_1, arg975_1, arg976_1, arg977_1, arg978_1, arg979_1, arg980_1, arg981_1, arg982_1, arg983_1, arg984_1, arg985_1, arg986_1, arg987_1, arg988_1, arg989_1, arg990_1, arg991_1, arg992_1, arg993_1, arg994_1, arg995_1, arg996_1, arg997_1, arg998_1, arg999_1, arg1000_1, arg1001_1, arg1002_1, arg1003_1, arg1004_1, arg1005_1, arg1006_1, arg1007_1, arg1008_1, arg1009_1, arg1010_1, arg1011_1, arg1012_1, arg1013_1, arg1014_1, arg1015_1, arg1016_1, arg1017_1, arg1018_1, arg1019_1, arg1020_1, arg1021_1, arg1022_1, arg1023_1, arg1024_1, arg1025_1, arg1026_1, arg1027_1, arg1028_1, arg1029_1, arg1030_1, arg1031_1, arg1032_1, arg1033_1, arg1034_1, arg1035_1, arg1036_1, arg1037_1, arg1038_1, arg1039_1, arg1040_1, arg1041_1, arg1042_1, arg1043_1, arg1044_1, arg1045_1, arg1046_1, arg1047_1, arg1048_1, arg1049_1, arg1050_1, arg1051_1, arg1052_1, arg1053_1, arg1054_1, arg1055_1, arg1056_1, arg1057_1, arg1058_1, arg1059_1, arg1060_1, arg1061_1, arg1062_1, arg1063_1, arg1064_1, arg1065_1, arg1066_1, arg1067_1, arg1068_1, arg1069_1, arg1070_1, arg1071_1, arg1072_1, arg1073_1, arg1074_1, arg1075_1, arg1076_1, arg1077_1, arg1078_1, arg1079_1, arg1080_1, arg1081_1, arg1082_1, arg1083_1, arg1084_1, arg1085_1, arg1086_1, arg1087_1, arg1088_1, arg1089_1, arg1090_1, arg1091_1, arg1092_1, arg1093_1, arg1094_1, arg1095_1, arg1096_1, arg1097_1, arg1098_1, arg1099_1, arg1100_1, arg1101_1, arg1102_1, arg1103_1, arg1104_1, arg1105_1, arg1106_1, arg1107_1, arg1108_1, arg1109_1, arg1110_1, arg1111_1, arg1112_1, arg1113_1, arg1114_1, arg1115_1, arg1116_1 = args
    args.clear()
    assert_size_stride(arg0_1, (512, ), (1, ))
    assert_size_stride(arg1_1, (512, ), (1, ))
    assert_size_stride(arg2_1, (128, ), (1, ))
    assert_size_stride(arg3_1, (128, ), (1, ))
    assert_size_stride(arg4_1, (128, ), (1, ))
    assert_size_stride(arg5_1, (128, ), (1, ))
    assert_size_stride(arg6_1, (128, ), (1, ))
    assert_size_stride(arg7_1, (128, ), (1, ))
    assert_size_stride(arg8_1, (128, ), (1, ))
    assert_size_stride(arg9_1, (128, ), (1, ))
    assert_size_stride(arg10_1, (128, ), (1, ))
    assert_size_stride(arg11_1, (128, ), (1, ))
    assert_size_stride(arg12_1, (128, ), (1, ))
    assert_size_stride(arg13_1, (128, ), (1, ))
    assert_size_stride(arg14_1, (128, ), (1, ))
    assert_size_stride(arg15_1, (128, ), (1, ))
    assert_size_stride(arg16_1, (512, ), (1, ))
    assert_size_stride(arg17_1, (512, ), (1, ))
    assert_size_stride(arg18_1, (128, ), (1, ))
    assert_size_stride(arg19_1, (128, ), (1, ))
    assert_size_stride(arg20_1, (128, ), (1, ))
    assert_size_stride(arg21_1, (128, ), (1, ))
    assert_size_stride(arg22_1, (128, ), (1, ))
    assert_size_stride(arg23_1, (128, ), (1, ))
    assert_size_stride(arg24_1, (128, ), (1, ))
    assert_size_stride(arg25_1, (128, ), (1, ))
    assert_size_stride(arg26_1, (128, ), (1, ))
    assert_size_stride(arg27_1, (128, ), (1, ))
    assert_size_stride(arg28_1, (128, ), (1, ))
    assert_size_stride(arg29_1, (128, ), (1, ))
    assert_size_stride(arg30_1, (128, ), (1, ))
    assert_size_stride(arg31_1, (128, ), (1, ))
    assert_size_stride(arg32_1, (512, ), (1, ))
    assert_size_stride(arg33_1, (512, ), (1, ))
    assert_size_stride(arg34_1, (128, ), (1, ))
    assert_size_stride(arg35_1, (128, ), (1, ))
    assert_size_stride(arg36_1, (128, ), (1, ))
    assert_size_stride(arg37_1, (128, ), (1, ))
    assert_size_stride(arg38_1, (128, ), (1, ))
    assert_size_stride(arg39_1, (128, ), (1, ))
    assert_size_stride(arg40_1, (128, ), (1, ))
    assert_size_stride(arg41_1, (128, ), (1, ))
    assert_size_stride(arg42_1, (128, ), (1, ))
    assert_size_stride(arg43_1, (128, ), (1, ))
    assert_size_stride(arg44_1, (128, ), (1, ))
    assert_size_stride(arg45_1, (128, ), (1, ))
    assert_size_stride(arg46_1, (128, ), (1, ))
    assert_size_stride(arg47_1, (128, ), (1, ))
    assert_size_stride(arg48_1, (512, ), (1, ))
    assert_size_stride(arg49_1, (512, ), (1, ))
    assert_size_stride(arg50_1, (128, ), (1, ))
    assert_size_stride(arg51_1, (128, ), (1, ))
    assert_size_stride(arg52_1, (128, ), (1, ))
    assert_size_stride(arg53_1, (128, ), (1, ))
    assert_size_stride(arg54_1, (128, ), (1, ))
    assert_size_stride(arg55_1, (128, ), (1, ))
    assert_size_stride(arg56_1, (128, ), (1, ))
    assert_size_stride(arg57_1, (128, ), (1, ))
    assert_size_stride(arg58_1, (128, ), (1, ))
    assert_size_stride(arg59_1, (128, ), (1, ))
    assert_size_stride(arg60_1, (128, ), (1, ))
    assert_size_stride(arg61_1, (128, ), (1, ))
    assert_size_stride(arg62_1, (128, ), (1, ))
    assert_size_stride(arg63_1, (128, ), (1, ))
    assert_size_stride(arg64_1, (512, ), (1, ))
    assert_size_stride(arg65_1, (512, ), (1, ))
    assert_size_stride(arg66_1, (128, ), (1, ))
    assert_size_stride(arg67_1, (128, ), (1, ))
    assert_size_stride(arg68_1, (128, ), (1, ))
    assert_size_stride(arg69_1, (128, ), (1, ))
    assert_size_stride(arg70_1, (128, ), (1, ))
    assert_size_stride(arg71_1, (128, ), (1, ))
    assert_size_stride(arg72_1, (128, ), (1, ))
    assert_size_stride(arg73_1, (128, ), (1, ))
    assert_size_stride(arg74_1, (128, ), (1, ))
    assert_size_stride(arg75_1, (128, ), (1, ))
    assert_size_stride(arg76_1, (128, ), (1, ))
    assert_size_stride(arg77_1, (128, ), (1, ))
    assert_size_stride(arg78_1, (128, ), (1, ))
    assert_size_stride(arg79_1, (128, ), (1, ))
    assert_size_stride(arg80_1, (512, ), (1, ))
    assert_size_stride(arg81_1, (512, ), (1, ))
    assert_size_stride(arg82_1, (128, ), (1, ))
    assert_size_stride(arg83_1, (128, ), (1, ))
    assert_size_stride(arg84_1, (128, ), (1, ))
    assert_size_stride(arg85_1, (128, ), (1, ))
    assert_size_stride(arg86_1, (128, ), (1, ))
    assert_size_stride(arg87_1, (128, ), (1, ))
    assert_size_stride(arg88_1, (128, ), (1, ))
    assert_size_stride(arg89_1, (128, ), (1, ))
    assert_size_stride(arg90_1, (128, ), (1, ))
    assert_size_stride(arg91_1, (128, ), (1, ))
    assert_size_stride(arg92_1, (128, ), (1, ))
    assert_size_stride(arg93_1, (128, ), (1, ))
    assert_size_stride(arg94_1, (128, ), (1, ))
    assert_size_stride(arg95_1, (128, ), (1, ))
    assert_size_stride(arg96_1, (512, ), (1, ))
    assert_size_stride(arg97_1, (512, ), (1, ))
    assert_size_stride(arg98_1, (128, ), (1, ))
    assert_size_stride(arg99_1, (128, ), (1, ))
    assert_size_stride(arg100_1, (128, ), (1, ))
    assert_size_stride(arg101_1, (128, ), (1, ))
    assert_size_stride(arg102_1, (128, ), (1, ))
    assert_size_stride(arg103_1, (128, ), (1, ))
    assert_size_stride(arg104_1, (128, ), (1, ))
    assert_size_stride(arg105_1, (128, ), (1, ))
    assert_size_stride(arg106_1, (128, ), (1, ))
    assert_size_stride(arg107_1, (128, ), (1, ))
    assert_size_stride(arg108_1, (128, ), (1, ))
    assert_size_stride(arg109_1, (128, ), (1, ))
    assert_size_stride(arg110_1, (128, ), (1, ))
    assert_size_stride(arg111_1, (128, ), (1, ))
    assert_size_stride(arg112_1, (512, ), (1, ))
    assert_size_stride(arg113_1, (512, ), (1, ))
    assert_size_stride(arg114_1, (128, ), (1, ))
    assert_size_stride(arg115_1, (128, ), (1, ))
    assert_size_stride(arg116_1, (128, ), (1, ))
    assert_size_stride(arg117_1, (128, ), (1, ))
    assert_size_stride(arg118_1, (128, ), (1, ))
    assert_size_stride(arg119_1, (128, ), (1, ))
    assert_size_stride(arg120_1, (128, ), (1, ))
    assert_size_stride(arg121_1, (128, ), (1, ))
    assert_size_stride(arg122_1, (128, ), (1, ))
    assert_size_stride(arg123_1, (128, ), (1, ))
    assert_size_stride(arg124_1, (128, ), (1, ))
    assert_size_stride(arg125_1, (128, ), (1, ))
    assert_size_stride(arg126_1, (128, ), (1, ))
    assert_size_stride(arg127_1, (128, ), (1, ))
    assert_size_stride(arg128_1, (512, ), (1, ))
    assert_size_stride(arg129_1, (512, ), (1, ))
    assert_size_stride(arg130_1, (128, ), (1, ))
    assert_size_stride(arg131_1, (128, ), (1, ))
    assert_size_stride(arg132_1, (128, ), (1, ))
    assert_size_stride(arg133_1, (128, ), (1, ))
    assert_size_stride(arg134_1, (128, ), (1, ))
    assert_size_stride(arg135_1, (128, ), (1, ))
    assert_size_stride(arg136_1, (128, ), (1, ))
    assert_size_stride(arg137_1, (128, ), (1, ))
    assert_size_stride(arg138_1, (128, ), (1, ))
    assert_size_stride(arg139_1, (128, ), (1, ))
    assert_size_stride(arg140_1, (128, ), (1, ))
    assert_size_stride(arg141_1, (128, ), (1, ))
    assert_size_stride(arg142_1, (128, ), (1, ))
    assert_size_stride(arg143_1, (128, ), (1, ))
    assert_size_stride(arg144_1, (512, ), (1, ))
    assert_size_stride(arg145_1, (512, ), (1, ))
    assert_size_stride(arg146_1, (128, ), (1, ))
    assert_size_stride(arg147_1, (128, ), (1, ))
    assert_size_stride(arg148_1, (128, ), (1, ))
    assert_size_stride(arg149_1, (128, ), (1, ))
    assert_size_stride(arg150_1, (128, ), (1, ))
    assert_size_stride(arg151_1, (128, ), (1, ))
    assert_size_stride(arg152_1, (128, ), (1, ))
    assert_size_stride(arg153_1, (128, ), (1, ))
    assert_size_stride(arg154_1, (128, ), (1, ))
    assert_size_stride(arg155_1, (128, ), (1, ))
    assert_size_stride(arg156_1, (128, ), (1, ))
    assert_size_stride(arg157_1, (128, ), (1, ))
    assert_size_stride(arg158_1, (128, ), (1, ))
    assert_size_stride(arg159_1, (128, ), (1, ))
    assert_size_stride(arg160_1, (512, ), (1, ))
    assert_size_stride(arg161_1, (512, ), (1, ))
    assert_size_stride(arg162_1, (128, ), (1, ))
    assert_size_stride(arg163_1, (128, ), (1, ))
    assert_size_stride(arg164_1, (128, ), (1, ))
    assert_size_stride(arg165_1, (128, ), (1, ))
    assert_size_stride(arg166_1, (128, ), (1, ))
    assert_size_stride(arg167_1, (128, ), (1, ))
    assert_size_stride(arg168_1, (128, ), (1, ))
    assert_size_stride(arg169_1, (128, ), (1, ))
    assert_size_stride(arg170_1, (128, ), (1, ))
    assert_size_stride(arg171_1, (128, ), (1, ))
    assert_size_stride(arg172_1, (128, ), (1, ))
    assert_size_stride(arg173_1, (128, ), (1, ))
    assert_size_stride(arg174_1, (128, ), (1, ))
    assert_size_stride(arg175_1, (128, ), (1, ))
    assert_size_stride(arg176_1, (512, ), (1, ))
    assert_size_stride(arg177_1, (512, ), (1, ))
    assert_size_stride(arg178_1, (128, ), (1, ))
    assert_size_stride(arg179_1, (128, ), (1, ))
    assert_size_stride(arg180_1, (128, ), (1, ))
    assert_size_stride(arg181_1, (128, ), (1, ))
    assert_size_stride(arg182_1, (128, ), (1, ))
    assert_size_stride(arg183_1, (128, ), (1, ))
    assert_size_stride(arg184_1, (128, ), (1, ))
    assert_size_stride(arg185_1, (128, ), (1, ))
    assert_size_stride(arg186_1, (128, ), (1, ))
    assert_size_stride(arg187_1, (128, ), (1, ))
    assert_size_stride(arg188_1, (128, ), (1, ))
    assert_size_stride(arg189_1, (128, ), (1, ))
    assert_size_stride(arg190_1, (128, ), (1, ))
    assert_size_stride(arg191_1, (128, ), (1, ))
    assert_size_stride(arg192_1, (512, ), (1, ))
    assert_size_stride(arg193_1, (512, ), (1, ))
    assert_size_stride(arg194_1, (128, ), (1, ))
    assert_size_stride(arg195_1, (128, ), (1, ))
    assert_size_stride(arg196_1, (128, ), (1, ))
    assert_size_stride(arg197_1, (128, ), (1, ))
    assert_size_stride(arg198_1, (128, ), (1, ))
    assert_size_stride(arg199_1, (128, ), (1, ))
    assert_size_stride(arg200_1, (128, ), (1, ))
    assert_size_stride(arg201_1, (128, ), (1, ))
    assert_size_stride(arg202_1, (128, ), (1, ))
    assert_size_stride(arg203_1, (128, ), (1, ))
    assert_size_stride(arg204_1, (128, ), (1, ))
    assert_size_stride(arg205_1, (128, ), (1, ))
    assert_size_stride(arg206_1, (128, ), (1, ))
    assert_size_stride(arg207_1, (128, ), (1, ))
    assert_size_stride(arg208_1, (512, ), (1, ))
    assert_size_stride(arg209_1, (512, ), (1, ))
    assert_size_stride(arg210_1, (128, ), (1, ))
    assert_size_stride(arg211_1, (128, ), (1, ))
    assert_size_stride(arg212_1, (128, ), (1, ))
    assert_size_stride(arg213_1, (128, ), (1, ))
    assert_size_stride(arg214_1, (128, ), (1, ))
    assert_size_stride(arg215_1, (128, ), (1, ))
    assert_size_stride(arg216_1, (128, ), (1, ))
    assert_size_stride(arg217_1, (128, ), (1, ))
    assert_size_stride(arg218_1, (128, ), (1, ))
    assert_size_stride(arg219_1, (128, ), (1, ))
    assert_size_stride(arg220_1, (128, ), (1, ))
    assert_size_stride(arg221_1, (128, ), (1, ))
    assert_size_stride(arg222_1, (128, ), (1, ))
    assert_size_stride(arg223_1, (128, ), (1, ))
    assert_size_stride(arg224_1, (512, ), (1, ))
    assert_size_stride(arg225_1, (512, ), (1, ))
    assert_size_stride(arg226_1, (128, ), (1, ))
    assert_size_stride(arg227_1, (128, ), (1, ))
    assert_size_stride(arg228_1, (128, ), (1, ))
    assert_size_stride(arg229_1, (128, ), (1, ))
    assert_size_stride(arg230_1, (128, ), (1, ))
    assert_size_stride(arg231_1, (128, ), (1, ))
    assert_size_stride(arg232_1, (128, ), (1, ))
    assert_size_stride(arg233_1, (128, ), (1, ))
    assert_size_stride(arg234_1, (128, ), (1, ))
    assert_size_stride(arg235_1, (128, ), (1, ))
    assert_size_stride(arg236_1, (128, ), (1, ))
    assert_size_stride(arg237_1, (128, ), (1, ))
    assert_size_stride(arg238_1, (128, ), (1, ))
    assert_size_stride(arg239_1, (128, ), (1, ))
    assert_size_stride(arg240_1, (512, ), (1, ))
    assert_size_stride(arg241_1, (512, ), (1, ))
    assert_size_stride(arg242_1, (128, ), (1, ))
    assert_size_stride(arg243_1, (128, ), (1, ))
    assert_size_stride(arg244_1, (128, ), (1, ))
    assert_size_stride(arg245_1, (128, ), (1, ))
    assert_size_stride(arg246_1, (128, ), (1, ))
    assert_size_stride(arg247_1, (128, ), (1, ))
    assert_size_stride(arg248_1, (128, ), (1, ))
    assert_size_stride(arg249_1, (128, ), (1, ))
    assert_size_stride(arg250_1, (128, ), (1, ))
    assert_size_stride(arg251_1, (128, ), (1, ))
    assert_size_stride(arg252_1, (128, ), (1, ))
    assert_size_stride(arg253_1, (128, ), (1, ))
    assert_size_stride(arg254_1, (128, ), (1, ))
    assert_size_stride(arg255_1, (128, ), (1, ))
    assert_size_stride(arg256_1, (512, ), (1, ))
    assert_size_stride(arg257_1, (512, ), (1, ))
    assert_size_stride(arg258_1, (128, ), (1, ))
    assert_size_stride(arg259_1, (128, ), (1, ))
    assert_size_stride(arg260_1, (128, ), (1, ))
    assert_size_stride(arg261_1, (128, ), (1, ))
    assert_size_stride(arg262_1, (128, ), (1, ))
    assert_size_stride(arg263_1, (128, ), (1, ))
    assert_size_stride(arg264_1, (128, ), (1, ))
    assert_size_stride(arg265_1, (128, ), (1, ))
    assert_size_stride(arg266_1, (128, ), (1, ))
    assert_size_stride(arg267_1, (128, ), (1, ))
    assert_size_stride(arg268_1, (128, ), (1, ))
    assert_size_stride(arg269_1, (128, ), (1, ))
    assert_size_stride(arg270_1, (128, ), (1, ))
    assert_size_stride(arg271_1, (128, ), (1, ))
    assert_size_stride(arg272_1, (512, ), (1, ))
    assert_size_stride(arg273_1, (512, ), (1, ))
    assert_size_stride(arg274_1, (128, ), (1, ))
    assert_size_stride(arg275_1, (128, ), (1, ))
    assert_size_stride(arg276_1, (128, ), (1, ))
    assert_size_stride(arg277_1, (128, ), (1, ))
    assert_size_stride(arg278_1, (128, ), (1, ))
    assert_size_stride(arg279_1, (128, ), (1, ))
    assert_size_stride(arg280_1, (128, ), (1, ))
    assert_size_stride(arg281_1, (128, ), (1, ))
    assert_size_stride(arg282_1, (128, ), (1, ))
    assert_size_stride(arg283_1, (128, ), (1, ))
    assert_size_stride(arg284_1, (128, ), (1, ))
    assert_size_stride(arg285_1, (128, ), (1, ))
    assert_size_stride(arg286_1, (128, ), (1, ))
    assert_size_stride(arg287_1, (128, ), (1, ))
    assert_size_stride(arg288_1, (512, ), (1, ))
    assert_size_stride(arg289_1, (512, ), (1, ))
    assert_size_stride(arg290_1, (128, ), (1, ))
    assert_size_stride(arg291_1, (128, ), (1, ))
    assert_size_stride(arg292_1, (128, ), (1, ))
    assert_size_stride(arg293_1, (128, ), (1, ))
    assert_size_stride(arg294_1, (128, ), (1, ))
    assert_size_stride(arg295_1, (128, ), (1, ))
    assert_size_stride(arg296_1, (128, ), (1, ))
    assert_size_stride(arg297_1, (128, ), (1, ))
    assert_size_stride(arg298_1, (128, ), (1, ))
    assert_size_stride(arg299_1, (128, ), (1, ))
    assert_size_stride(arg300_1, (128, ), (1, ))
    assert_size_stride(arg301_1, (128, ), (1, ))
    assert_size_stride(arg302_1, (128, ), (1, ))
    assert_size_stride(arg303_1, (128, ), (1, ))
    assert_size_stride(arg304_1, (512, ), (1, ))
    assert_size_stride(arg305_1, (512, ), (1, ))
    assert_size_stride(arg306_1, (128, ), (1, ))
    assert_size_stride(arg307_1, (128, ), (1, ))
    assert_size_stride(arg308_1, (128, ), (1, ))
    assert_size_stride(arg309_1, (128, ), (1, ))
    assert_size_stride(arg310_1, (128, ), (1, ))
    assert_size_stride(arg311_1, (128, ), (1, ))
    assert_size_stride(arg312_1, (128, ), (1, ))
    assert_size_stride(arg313_1, (128, ), (1, ))
    assert_size_stride(arg314_1, (128, ), (1, ))
    assert_size_stride(arg315_1, (128, ), (1, ))
    assert_size_stride(arg316_1, (128, ), (1, ))
    assert_size_stride(arg317_1, (128, ), (1, ))
    assert_size_stride(arg318_1, (128, ), (1, ))
    assert_size_stride(arg319_1, (128, ), (1, ))
    assert_size_stride(arg320_1, (512, ), (1, ))
    assert_size_stride(arg321_1, (512, ), (1, ))
    assert_size_stride(arg322_1, (128, ), (1, ))
    assert_size_stride(arg323_1, (128, ), (1, ))
    assert_size_stride(arg324_1, (128, ), (1, ))
    assert_size_stride(arg325_1, (128, ), (1, ))
    assert_size_stride(arg326_1, (128, ), (1, ))
    assert_size_stride(arg327_1, (128, ), (1, ))
    assert_size_stride(arg328_1, (128, ), (1, ))
    assert_size_stride(arg329_1, (128, ), (1, ))
    assert_size_stride(arg330_1, (128, ), (1, ))
    assert_size_stride(arg331_1, (128, ), (1, ))
    assert_size_stride(arg332_1, (128, ), (1, ))
    assert_size_stride(arg333_1, (128, ), (1, ))
    assert_size_stride(arg334_1, (128, ), (1, ))
    assert_size_stride(arg335_1, (128, ), (1, ))
    assert_size_stride(arg336_1, (512, ), (1, ))
    assert_size_stride(arg337_1, (512, ), (1, ))
    assert_size_stride(arg338_1, (128, ), (1, ))
    assert_size_stride(arg339_1, (128, ), (1, ))
    assert_size_stride(arg340_1, (128, ), (1, ))
    assert_size_stride(arg341_1, (128, ), (1, ))
    assert_size_stride(arg342_1, (128, ), (1, ))
    assert_size_stride(arg343_1, (128, ), (1, ))
    assert_size_stride(arg344_1, (128, ), (1, ))
    assert_size_stride(arg345_1, (128, ), (1, ))
    assert_size_stride(arg346_1, (128, ), (1, ))
    assert_size_stride(arg347_1, (128, ), (1, ))
    assert_size_stride(arg348_1, (128, ), (1, ))
    assert_size_stride(arg349_1, (128, ), (1, ))
    assert_size_stride(arg350_1, (128, ), (1, ))
    assert_size_stride(arg351_1, (128, ), (1, ))
    assert_size_stride(arg352_1, (512, ), (1, ))
    assert_size_stride(arg353_1, (512, ), (1, ))
    assert_size_stride(arg354_1, (128, ), (1, ))
    assert_size_stride(arg355_1, (128, ), (1, ))
    assert_size_stride(arg356_1, (128, ), (1, ))
    assert_size_stride(arg357_1, (128, ), (1, ))
    assert_size_stride(arg358_1, (128, ), (1, ))
    assert_size_stride(arg359_1, (128, ), (1, ))
    assert_size_stride(arg360_1, (128, ), (1, ))
    assert_size_stride(arg361_1, (128, ), (1, ))
    assert_size_stride(arg362_1, (128, ), (1, ))
    assert_size_stride(arg363_1, (128, ), (1, ))
    assert_size_stride(arg364_1, (128, ), (1, ))
    assert_size_stride(arg365_1, (128, ), (1, ))
    assert_size_stride(arg366_1, (128, ), (1, ))
    assert_size_stride(arg367_1, (128, ), (1, ))
    assert_size_stride(arg368_1, (512, ), (1, ))
    assert_size_stride(arg369_1, (512, ), (1, ))
    assert_size_stride(arg370_1, (128, ), (1, ))
    assert_size_stride(arg371_1, (128, ), (1, ))
    assert_size_stride(arg372_1, (128, ), (1, ))
    assert_size_stride(arg373_1, (128, ), (1, ))
    assert_size_stride(arg374_1, (128, ), (1, ))
    assert_size_stride(arg375_1, (128, ), (1, ))
    assert_size_stride(arg376_1, (128, ), (1, ))
    assert_size_stride(arg377_1, (128, ), (1, ))
    assert_size_stride(arg378_1, (128, ), (1, ))
    assert_size_stride(arg379_1, (128, ), (1, ))
    assert_size_stride(arg380_1, (128, ), (1, ))
    assert_size_stride(arg381_1, (128, ), (1, ))
    assert_size_stride(arg382_1, (128, ), (1, ))
    assert_size_stride(arg383_1, (128, ), (1, ))
    assert_size_stride(arg384_1, (512, ), (1, ))
    assert_size_stride(arg385_1, (512, ), (1, ))
    assert_size_stride(arg386_1, (30522, 128), (128, 1))
    assert_size_stride(arg387_1, (512, 384), (384, 1))
    assert_size_stride(arg388_1, (512, ), (1, ))
    assert_size_stride(arg389_1, (512, 512), (512, 1))
    assert_size_stride(arg390_1, (2, 512), (512, 1))
    assert_size_stride(arg391_1, (128, 512), (512, 1))
    assert_size_stride(arg392_1, (128, ), (1, ))
    assert_size_stride(arg393_1, (128, 512), (512, 1))
    assert_size_stride(arg394_1, (128, ), (1, ))
    assert_size_stride(arg395_1, (128, 128), (128, 1))
    assert_size_stride(arg396_1, (128, ), (1, ))
    assert_size_stride(arg397_1, (128, 128), (128, 1))
    assert_size_stride(arg398_1, (128, ), (1, ))
    assert_size_stride(arg399_1, (128, 512), (512, 1))
    assert_size_stride(arg400_1, (128, ), (1, ))
    assert_size_stride(arg401_1, (128, 128), (128, 1))
    assert_size_stride(arg402_1, (128, ), (1, ))
    assert_size_stride(arg403_1, (512, 128), (128, 1))
    assert_size_stride(arg404_1, (512, ), (1, ))
    assert_size_stride(arg405_1, (128, 512), (512, 1))
    assert_size_stride(arg406_1, (128, ), (1, ))
    assert_size_stride(arg407_1, (512, 128), (128, 1))
    assert_size_stride(arg408_1, (512, ), (1, ))
    assert_size_stride(arg409_1, (128, 512), (512, 1))
    assert_size_stride(arg410_1, (128, ), (1, ))
    assert_size_stride(arg411_1, (512, 128), (128, 1))
    assert_size_stride(arg412_1, (512, ), (1, ))
    assert_size_stride(arg413_1, (128, 512), (512, 1))
    assert_size_stride(arg414_1, (128, ), (1, ))
    assert_size_stride(arg415_1, (512, 128), (128, 1))
    assert_size_stride(arg416_1, (512, ), (1, ))
    assert_size_stride(arg417_1, (128, 512), (512, 1))
    assert_size_stride(arg418_1, (128, ), (1, ))
    assert_size_stride(arg419_1, (512, 128), (128, 1))
    assert_size_stride(arg420_1, (512, ), (1, ))
    assert_size_stride(arg421_1, (128, 512), (512, 1))
    assert_size_stride(arg422_1, (128, ), (1, ))
    assert_size_stride(arg423_1, (128, 512), (512, 1))
    assert_size_stride(arg424_1, (128, ), (1, ))
    assert_size_stride(arg425_1, (128, 128), (128, 1))
    assert_size_stride(arg426_1, (128, ), (1, ))
    assert_size_stride(arg427_1, (128, 128), (128, 1))
    assert_size_stride(arg428_1, (128, ), (1, ))
    assert_size_stride(arg429_1, (128, 512), (512, 1))
    assert_size_stride(arg430_1, (128, ), (1, ))
    assert_size_stride(arg431_1, (128, 128), (128, 1))
    assert_size_stride(arg432_1, (128, ), (1, ))
    assert_size_stride(arg433_1, (512, 128), (128, 1))
    assert_size_stride(arg434_1, (512, ), (1, ))
    assert_size_stride(arg435_1, (128, 512), (512, 1))
    assert_size_stride(arg436_1, (128, ), (1, ))
    assert_size_stride(arg437_1, (512, 128), (128, 1))
    assert_size_stride(arg438_1, (512, ), (1, ))
    assert_size_stride(arg439_1, (128, 512), (512, 1))
    assert_size_stride(arg440_1, (128, ), (1, ))
    assert_size_stride(arg441_1, (512, 128), (128, 1))
    assert_size_stride(arg442_1, (512, ), (1, ))
    assert_size_stride(arg443_1, (128, 512), (512, 1))
    assert_size_stride(arg444_1, (128, ), (1, ))
    assert_size_stride(arg445_1, (512, 128), (128, 1))
    assert_size_stride(arg446_1, (512, ), (1, ))
    assert_size_stride(arg447_1, (128, 512), (512, 1))
    assert_size_stride(arg448_1, (128, ), (1, ))
    assert_size_stride(arg449_1, (512, 128), (128, 1))
    assert_size_stride(arg450_1, (512, ), (1, ))
    assert_size_stride(arg451_1, (128, 512), (512, 1))
    assert_size_stride(arg452_1, (128, ), (1, ))
    assert_size_stride(arg453_1, (128, 512), (512, 1))
    assert_size_stride(arg454_1, (128, ), (1, ))
    assert_size_stride(arg455_1, (128, 128), (128, 1))
    assert_size_stride(arg456_1, (128, ), (1, ))
    assert_size_stride(arg457_1, (128, 128), (128, 1))
    assert_size_stride(arg458_1, (128, ), (1, ))
    assert_size_stride(arg459_1, (128, 512), (512, 1))
    assert_size_stride(arg460_1, (128, ), (1, ))
    assert_size_stride(arg461_1, (128, 128), (128, 1))
    assert_size_stride(arg462_1, (128, ), (1, ))
    assert_size_stride(arg463_1, (512, 128), (128, 1))
    assert_size_stride(arg464_1, (512, ), (1, ))
    assert_size_stride(arg465_1, (128, 512), (512, 1))
    assert_size_stride(arg466_1, (128, ), (1, ))
    assert_size_stride(arg467_1, (512, 128), (128, 1))
    assert_size_stride(arg468_1, (512, ), (1, ))
    assert_size_stride(arg469_1, (128, 512), (512, 1))
    assert_size_stride(arg470_1, (128, ), (1, ))
    assert_size_stride(arg471_1, (512, 128), (128, 1))
    assert_size_stride(arg472_1, (512, ), (1, ))
    assert_size_stride(arg473_1, (128, 512), (512, 1))
    assert_size_stride(arg474_1, (128, ), (1, ))
    assert_size_stride(arg475_1, (512, 128), (128, 1))
    assert_size_stride(arg476_1, (512, ), (1, ))
    assert_size_stride(arg477_1, (128, 512), (512, 1))
    assert_size_stride(arg478_1, (128, ), (1, ))
    assert_size_stride(arg479_1, (512, 128), (128, 1))
    assert_size_stride(arg480_1, (512, ), (1, ))
    assert_size_stride(arg481_1, (128, 512), (512, 1))
    assert_size_stride(arg482_1, (128, ), (1, ))
    assert_size_stride(arg483_1, (128, 512), (512, 1))
    assert_size_stride(arg484_1, (128, ), (1, ))
    assert_size_stride(arg485_1, (128, 128), (128, 1))
    assert_size_stride(arg486_1, (128, ), (1, ))
    assert_size_stride(arg487_1, (128, 128), (128, 1))
    assert_size_stride(arg488_1, (128, ), (1, ))
    assert_size_stride(arg489_1, (128, 512), (512, 1))
    assert_size_stride(arg490_1, (128, ), (1, ))
    assert_size_stride(arg491_1, (128, 128), (128, 1))
    assert_size_stride(arg492_1, (128, ), (1, ))
    assert_size_stride(arg493_1, (512, 128), (128, 1))
    assert_size_stride(arg494_1, (512, ), (1, ))
    assert_size_stride(arg495_1, (128, 512), (512, 1))
    assert_size_stride(arg496_1, (128, ), (1, ))
    assert_size_stride(arg497_1, (512, 128), (128, 1))
    assert_size_stride(arg498_1, (512, ), (1, ))
    assert_size_stride(arg499_1, (128, 512), (512, 1))
    assert_size_stride(arg500_1, (128, ), (1, ))
    assert_size_stride(arg501_1, (512, 128), (128, 1))
    assert_size_stride(arg502_1, (512, ), (1, ))
    assert_size_stride(arg503_1, (128, 512), (512, 1))
    assert_size_stride(arg504_1, (128, ), (1, ))
    assert_size_stride(arg505_1, (512, 128), (128, 1))
    assert_size_stride(arg506_1, (512, ), (1, ))
    assert_size_stride(arg507_1, (128, 512), (512, 1))
    assert_size_stride(arg508_1, (128, ), (1, ))
    assert_size_stride(arg509_1, (512, 128), (128, 1))
    assert_size_stride(arg510_1, (512, ), (1, ))
    assert_size_stride(arg511_1, (128, 512), (512, 1))
    assert_size_stride(arg512_1, (128, ), (1, ))
    assert_size_stride(arg513_1, (128, 512), (512, 1))
    assert_size_stride(arg514_1, (128, ), (1, ))
    assert_size_stride(arg515_1, (128, 128), (128, 1))
    assert_size_stride(arg516_1, (128, ), (1, ))
    assert_size_stride(arg517_1, (128, 128), (128, 1))
    assert_size_stride(arg518_1, (128, ), (1, ))
    assert_size_stride(arg519_1, (128, 512), (512, 1))
    assert_size_stride(arg520_1, (128, ), (1, ))
    assert_size_stride(arg521_1, (128, 128), (128, 1))
    assert_size_stride(arg522_1, (128, ), (1, ))
    assert_size_stride(arg523_1, (512, 128), (128, 1))
    assert_size_stride(arg524_1, (512, ), (1, ))
    assert_size_stride(arg525_1, (128, 512), (512, 1))
    assert_size_stride(arg526_1, (128, ), (1, ))
    assert_size_stride(arg527_1, (512, 128), (128, 1))
    assert_size_stride(arg528_1, (512, ), (1, ))
    assert_size_stride(arg529_1, (128, 512), (512, 1))
    assert_size_stride(arg530_1, (128, ), (1, ))
    assert_size_stride(arg531_1, (512, 128), (128, 1))
    assert_size_stride(arg532_1, (512, ), (1, ))
    assert_size_stride(arg533_1, (128, 512), (512, 1))
    assert_size_stride(arg534_1, (128, ), (1, ))
    assert_size_stride(arg535_1, (512, 128), (128, 1))
    assert_size_stride(arg536_1, (512, ), (1, ))
    assert_size_stride(arg537_1, (128, 512), (512, 1))
    assert_size_stride(arg538_1, (128, ), (1, ))
    assert_size_stride(arg539_1, (512, 128), (128, 1))
    assert_size_stride(arg540_1, (512, ), (1, ))
    assert_size_stride(arg541_1, (128, 512), (512, 1))
    assert_size_stride(arg542_1, (128, ), (1, ))
    assert_size_stride(arg543_1, (128, 512), (512, 1))
    assert_size_stride(arg544_1, (128, ), (1, ))
    assert_size_stride(arg545_1, (128, 128), (128, 1))
    assert_size_stride(arg546_1, (128, ), (1, ))
    assert_size_stride(arg547_1, (128, 128), (128, 1))
    assert_size_stride(arg548_1, (128, ), (1, ))
    assert_size_stride(arg549_1, (128, 512), (512, 1))
    assert_size_stride(arg550_1, (128, ), (1, ))
    assert_size_stride(arg551_1, (128, 128), (128, 1))
    assert_size_stride(arg552_1, (128, ), (1, ))
    assert_size_stride(arg553_1, (512, 128), (128, 1))
    assert_size_stride(arg554_1, (512, ), (1, ))
    assert_size_stride(arg555_1, (128, 512), (512, 1))
    assert_size_stride(arg556_1, (128, ), (1, ))
    assert_size_stride(arg557_1, (512, 128), (128, 1))
    assert_size_stride(arg558_1, (512, ), (1, ))
    assert_size_stride(arg559_1, (128, 512), (512, 1))
    assert_size_stride(arg560_1, (128, ), (1, ))
    assert_size_stride(arg561_1, (512, 128), (128, 1))
    assert_size_stride(arg562_1, (512, ), (1, ))
    assert_size_stride(arg563_1, (128, 512), (512, 1))
    assert_size_stride(arg564_1, (128, ), (1, ))
    assert_size_stride(arg565_1, (512, 128), (128, 1))
    assert_size_stride(arg566_1, (512, ), (1, ))
    assert_size_stride(arg567_1, (128, 512), (512, 1))
    assert_size_stride(arg568_1, (128, ), (1, ))
    assert_size_stride(arg569_1, (512, 128), (128, 1))
    assert_size_stride(arg570_1, (512, ), (1, ))
    assert_size_stride(arg571_1, (128, 512), (512, 1))
    assert_size_stride(arg572_1, (128, ), (1, ))
    assert_size_stride(arg573_1, (128, 512), (512, 1))
    assert_size_stride(arg574_1, (128, ), (1, ))
    assert_size_stride(arg575_1, (128, 128), (128, 1))
    assert_size_stride(arg576_1, (128, ), (1, ))
    assert_size_stride(arg577_1, (128, 128), (128, 1))
    assert_size_stride(arg578_1, (128, ), (1, ))
    assert_size_stride(arg579_1, (128, 512), (512, 1))
    assert_size_stride(arg580_1, (128, ), (1, ))
    assert_size_stride(arg581_1, (128, 128), (128, 1))
    assert_size_stride(arg582_1, (128, ), (1, ))
    assert_size_stride(arg583_1, (512, 128), (128, 1))
    assert_size_stride(arg584_1, (512, ), (1, ))
    assert_size_stride(arg585_1, (128, 512), (512, 1))
    assert_size_stride(arg586_1, (128, ), (1, ))
    assert_size_stride(arg587_1, (512, 128), (128, 1))
    assert_size_stride(arg588_1, (512, ), (1, ))
    assert_size_stride(arg589_1, (128, 512), (512, 1))
    assert_size_stride(arg590_1, (128, ), (1, ))
    assert_size_stride(arg591_1, (512, 128), (128, 1))
    assert_size_stride(arg592_1, (512, ), (1, ))
    assert_size_stride(arg593_1, (128, 512), (512, 1))
    assert_size_stride(arg594_1, (128, ), (1, ))
    assert_size_stride(arg595_1, (512, 128), (128, 1))
    assert_size_stride(arg596_1, (512, ), (1, ))
    assert_size_stride(arg597_1, (128, 512), (512, 1))
    assert_size_stride(arg598_1, (128, ), (1, ))
    assert_size_stride(arg599_1, (512, 128), (128, 1))
    assert_size_stride(arg600_1, (512, ), (1, ))
    assert_size_stride(arg601_1, (128, 512), (512, 1))
    assert_size_stride(arg602_1, (128, ), (1, ))
    assert_size_stride(arg603_1, (128, 512), (512, 1))
    assert_size_stride(arg604_1, (128, ), (1, ))
    assert_size_stride(arg605_1, (128, 128), (128, 1))
    assert_size_stride(arg606_1, (128, ), (1, ))
    assert_size_stride(arg607_1, (128, 128), (128, 1))
    assert_size_stride(arg608_1, (128, ), (1, ))
    assert_size_stride(arg609_1, (128, 512), (512, 1))
    assert_size_stride(arg610_1, (128, ), (1, ))
    assert_size_stride(arg611_1, (128, 128), (128, 1))
    assert_size_stride(arg612_1, (128, ), (1, ))
    assert_size_stride(arg613_1, (512, 128), (128, 1))
    assert_size_stride(arg614_1, (512, ), (1, ))
    assert_size_stride(arg615_1, (128, 512), (512, 1))
    assert_size_stride(arg616_1, (128, ), (1, ))
    assert_size_stride(arg617_1, (512, 128), (128, 1))
    assert_size_stride(arg618_1, (512, ), (1, ))
    assert_size_stride(arg619_1, (128, 512), (512, 1))
    assert_size_stride(arg620_1, (128, ), (1, ))
    assert_size_stride(arg621_1, (512, 128), (128, 1))
    assert_size_stride(arg622_1, (512, ), (1, ))
    assert_size_stride(arg623_1, (128, 512), (512, 1))
    assert_size_stride(arg624_1, (128, ), (1, ))
    assert_size_stride(arg625_1, (512, 128), (128, 1))
    assert_size_stride(arg626_1, (512, ), (1, ))
    assert_size_stride(arg627_1, (128, 512), (512, 1))
    assert_size_stride(arg628_1, (128, ), (1, ))
    assert_size_stride(arg629_1, (512, 128), (128, 1))
    assert_size_stride(arg630_1, (512, ), (1, ))
    assert_size_stride(arg631_1, (128, 512), (512, 1))
    assert_size_stride(arg632_1, (128, ), (1, ))
    assert_size_stride(arg633_1, (128, 512), (512, 1))
    assert_size_stride(arg634_1, (128, ), (1, ))
    assert_size_stride(arg635_1, (128, 128), (128, 1))
    assert_size_stride(arg636_1, (128, ), (1, ))
    assert_size_stride(arg637_1, (128, 128), (128, 1))
    assert_size_stride(arg638_1, (128, ), (1, ))
    assert_size_stride(arg639_1, (128, 512), (512, 1))
    assert_size_stride(arg640_1, (128, ), (1, ))
    assert_size_stride(arg641_1, (128, 128), (128, 1))
    assert_size_stride(arg642_1, (128, ), (1, ))
    assert_size_stride(arg643_1, (512, 128), (128, 1))
    assert_size_stride(arg644_1, (512, ), (1, ))
    assert_size_stride(arg645_1, (128, 512), (512, 1))
    assert_size_stride(arg646_1, (128, ), (1, ))
    assert_size_stride(arg647_1, (512, 128), (128, 1))
    assert_size_stride(arg648_1, (512, ), (1, ))
    assert_size_stride(arg649_1, (128, 512), (512, 1))
    assert_size_stride(arg650_1, (128, ), (1, ))
    assert_size_stride(arg651_1, (512, 128), (128, 1))
    assert_size_stride(arg652_1, (512, ), (1, ))
    assert_size_stride(arg653_1, (128, 512), (512, 1))
    assert_size_stride(arg654_1, (128, ), (1, ))
    assert_size_stride(arg655_1, (512, 128), (128, 1))
    assert_size_stride(arg656_1, (512, ), (1, ))
    assert_size_stride(arg657_1, (128, 512), (512, 1))
    assert_size_stride(arg658_1, (128, ), (1, ))
    assert_size_stride(arg659_1, (512, 128), (128, 1))
    assert_size_stride(arg660_1, (512, ), (1, ))
    assert_size_stride(arg661_1, (128, 512), (512, 1))
    assert_size_stride(arg662_1, (128, ), (1, ))
    assert_size_stride(arg663_1, (128, 512), (512, 1))
    assert_size_stride(arg664_1, (128, ), (1, ))
    assert_size_stride(arg665_1, (128, 128), (128, 1))
    assert_size_stride(arg666_1, (128, ), (1, ))
    assert_size_stride(arg667_1, (128, 128), (128, 1))
    assert_size_stride(arg668_1, (128, ), (1, ))
    assert_size_stride(arg669_1, (128, 512), (512, 1))
    assert_size_stride(arg670_1, (128, ), (1, ))
    assert_size_stride(arg671_1, (128, 128), (128, 1))
    assert_size_stride(arg672_1, (128, ), (1, ))
    assert_size_stride(arg673_1, (512, 128), (128, 1))
    assert_size_stride(arg674_1, (512, ), (1, ))
    assert_size_stride(arg675_1, (128, 512), (512, 1))
    assert_size_stride(arg676_1, (128, ), (1, ))
    assert_size_stride(arg677_1, (512, 128), (128, 1))
    assert_size_stride(arg678_1, (512, ), (1, ))
    assert_size_stride(arg679_1, (128, 512), (512, 1))
    assert_size_stride(arg680_1, (128, ), (1, ))
    assert_size_stride(arg681_1, (512, 128), (128, 1))
    assert_size_stride(arg682_1, (512, ), (1, ))
    assert_size_stride(arg683_1, (128, 512), (512, 1))
    assert_size_stride(arg684_1, (128, ), (1, ))
    assert_size_stride(arg685_1, (512, 128), (128, 1))
    assert_size_stride(arg686_1, (512, ), (1, ))
    assert_size_stride(arg687_1, (128, 512), (512, 1))
    assert_size_stride(arg688_1, (128, ), (1, ))
    assert_size_stride(arg689_1, (512, 128), (128, 1))
    assert_size_stride(arg690_1, (512, ), (1, ))
    assert_size_stride(arg691_1, (128, 512), (512, 1))
    assert_size_stride(arg692_1, (128, ), (1, ))
    assert_size_stride(arg693_1, (128, 512), (512, 1))
    assert_size_stride(arg694_1, (128, ), (1, ))
    assert_size_stride(arg695_1, (128, 128), (128, 1))
    assert_size_stride(arg696_1, (128, ), (1, ))
    assert_size_stride(arg697_1, (128, 128), (128, 1))
    assert_size_stride(arg698_1, (128, ), (1, ))
    assert_size_stride(arg699_1, (128, 512), (512, 1))
    assert_size_stride(arg700_1, (128, ), (1, ))
    assert_size_stride(arg701_1, (128, 128), (128, 1))
    assert_size_stride(arg702_1, (128, ), (1, ))
    assert_size_stride(arg703_1, (512, 128), (128, 1))
    assert_size_stride(arg704_1, (512, ), (1, ))
    assert_size_stride(arg705_1, (128, 512), (512, 1))
    assert_size_stride(arg706_1, (128, ), (1, ))
    assert_size_stride(arg707_1, (512, 128), (128, 1))
    assert_size_stride(arg708_1, (512, ), (1, ))
    assert_size_stride(arg709_1, (128, 512), (512, 1))
    assert_size_stride(arg710_1, (128, ), (1, ))
    assert_size_stride(arg711_1, (512, 128), (128, 1))
    assert_size_stride(arg712_1, (512, ), (1, ))
    assert_size_stride(arg713_1, (128, 512), (512, 1))
    assert_size_stride(arg714_1, (128, ), (1, ))
    assert_size_stride(arg715_1, (512, 128), (128, 1))
    assert_size_stride(arg716_1, (512, ), (1, ))
    assert_size_stride(arg717_1, (128, 512), (512, 1))
    assert_size_stride(arg718_1, (128, ), (1, ))
    assert_size_stride(arg719_1, (512, 128), (128, 1))
    assert_size_stride(arg720_1, (512, ), (1, ))
    assert_size_stride(arg721_1, (128, 512), (512, 1))
    assert_size_stride(arg722_1, (128, ), (1, ))
    assert_size_stride(arg723_1, (128, 512), (512, 1))
    assert_size_stride(arg724_1, (128, ), (1, ))
    assert_size_stride(arg725_1, (128, 128), (128, 1))
    assert_size_stride(arg726_1, (128, ), (1, ))
    assert_size_stride(arg727_1, (128, 128), (128, 1))
    assert_size_stride(arg728_1, (128, ), (1, ))
    assert_size_stride(arg729_1, (128, 512), (512, 1))
    assert_size_stride(arg730_1, (128, ), (1, ))
    assert_size_stride(arg731_1, (128, 128), (128, 1))
    assert_size_stride(arg732_1, (128, ), (1, ))
    assert_size_stride(arg733_1, (512, 128), (128, 1))
    assert_size_stride(arg734_1, (512, ), (1, ))
    assert_size_stride(arg735_1, (128, 512), (512, 1))
    assert_size_stride(arg736_1, (128, ), (1, ))
    assert_size_stride(arg737_1, (512, 128), (128, 1))
    assert_size_stride(arg738_1, (512, ), (1, ))
    assert_size_stride(arg739_1, (128, 512), (512, 1))
    assert_size_stride(arg740_1, (128, ), (1, ))
    assert_size_stride(arg741_1, (512, 128), (128, 1))
    assert_size_stride(arg742_1, (512, ), (1, ))
    assert_size_stride(arg743_1, (128, 512), (512, 1))
    assert_size_stride(arg744_1, (128, ), (1, ))
    assert_size_stride(arg745_1, (512, 128), (128, 1))
    assert_size_stride(arg746_1, (512, ), (1, ))
    assert_size_stride(arg747_1, (128, 512), (512, 1))
    assert_size_stride(arg748_1, (128, ), (1, ))
    assert_size_stride(arg749_1, (512, 128), (128, 1))
    assert_size_stride(arg750_1, (512, ), (1, ))
    assert_size_stride(arg751_1, (128, 512), (512, 1))
    assert_size_stride(arg752_1, (128, ), (1, ))
    assert_size_stride(arg753_1, (128, 512), (512, 1))
    assert_size_stride(arg754_1, (128, ), (1, ))
    assert_size_stride(arg755_1, (128, 128), (128, 1))
    assert_size_stride(arg756_1, (128, ), (1, ))
    assert_size_stride(arg757_1, (128, 128), (128, 1))
    assert_size_stride(arg758_1, (128, ), (1, ))
    assert_size_stride(arg759_1, (128, 512), (512, 1))
    assert_size_stride(arg760_1, (128, ), (1, ))
    assert_size_stride(arg761_1, (128, 128), (128, 1))
    assert_size_stride(arg762_1, (128, ), (1, ))
    assert_size_stride(arg763_1, (512, 128), (128, 1))
    assert_size_stride(arg764_1, (512, ), (1, ))
    assert_size_stride(arg765_1, (128, 512), (512, 1))
    assert_size_stride(arg766_1, (128, ), (1, ))
    assert_size_stride(arg767_1, (512, 128), (128, 1))
    assert_size_stride(arg768_1, (512, ), (1, ))
    assert_size_stride(arg769_1, (128, 512), (512, 1))
    assert_size_stride(arg770_1, (128, ), (1, ))
    assert_size_stride(arg771_1, (512, 128), (128, 1))
    assert_size_stride(arg772_1, (512, ), (1, ))
    assert_size_stride(arg773_1, (128, 512), (512, 1))
    assert_size_stride(arg774_1, (128, ), (1, ))
    assert_size_stride(arg775_1, (512, 128), (128, 1))
    assert_size_stride(arg776_1, (512, ), (1, ))
    assert_size_stride(arg777_1, (128, 512), (512, 1))
    assert_size_stride(arg778_1, (128, ), (1, ))
    assert_size_stride(arg779_1, (512, 128), (128, 1))
    assert_size_stride(arg780_1, (512, ), (1, ))
    assert_size_stride(arg781_1, (128, 512), (512, 1))
    assert_size_stride(arg782_1, (128, ), (1, ))
    assert_size_stride(arg783_1, (128, 512), (512, 1))
    assert_size_stride(arg784_1, (128, ), (1, ))
    assert_size_stride(arg785_1, (128, 128), (128, 1))
    assert_size_stride(arg786_1, (128, ), (1, ))
    assert_size_stride(arg787_1, (128, 128), (128, 1))
    assert_size_stride(arg788_1, (128, ), (1, ))
    assert_size_stride(arg789_1, (128, 512), (512, 1))
    assert_size_stride(arg790_1, (128, ), (1, ))
    assert_size_stride(arg791_1, (128, 128), (128, 1))
    assert_size_stride(arg792_1, (128, ), (1, ))
    assert_size_stride(arg793_1, (512, 128), (128, 1))
    assert_size_stride(arg794_1, (512, ), (1, ))
    assert_size_stride(arg795_1, (128, 512), (512, 1))
    assert_size_stride(arg796_1, (128, ), (1, ))
    assert_size_stride(arg797_1, (512, 128), (128, 1))
    assert_size_stride(arg798_1, (512, ), (1, ))
    assert_size_stride(arg799_1, (128, 512), (512, 1))
    assert_size_stride(arg800_1, (128, ), (1, ))
    assert_size_stride(arg801_1, (512, 128), (128, 1))
    assert_size_stride(arg802_1, (512, ), (1, ))
    assert_size_stride(arg803_1, (128, 512), (512, 1))
    assert_size_stride(arg804_1, (128, ), (1, ))
    assert_size_stride(arg805_1, (512, 128), (128, 1))
    assert_size_stride(arg806_1, (512, ), (1, ))
    assert_size_stride(arg807_1, (128, 512), (512, 1))
    assert_size_stride(arg808_1, (128, ), (1, ))
    assert_size_stride(arg809_1, (512, 128), (128, 1))
    assert_size_stride(arg810_1, (512, ), (1, ))
    assert_size_stride(arg811_1, (128, 512), (512, 1))
    assert_size_stride(arg812_1, (128, ), (1, ))
    assert_size_stride(arg813_1, (128, 512), (512, 1))
    assert_size_stride(arg814_1, (128, ), (1, ))
    assert_size_stride(arg815_1, (128, 128), (128, 1))
    assert_size_stride(arg816_1, (128, ), (1, ))
    assert_size_stride(arg817_1, (128, 128), (128, 1))
    assert_size_stride(arg818_1, (128, ), (1, ))
    assert_size_stride(arg819_1, (128, 512), (512, 1))
    assert_size_stride(arg820_1, (128, ), (1, ))
    assert_size_stride(arg821_1, (128, 128), (128, 1))
    assert_size_stride(arg822_1, (128, ), (1, ))
    assert_size_stride(arg823_1, (512, 128), (128, 1))
    assert_size_stride(arg824_1, (512, ), (1, ))
    assert_size_stride(arg825_1, (128, 512), (512, 1))
    assert_size_stride(arg826_1, (128, ), (1, ))
    assert_size_stride(arg827_1, (512, 128), (128, 1))
    assert_size_stride(arg828_1, (512, ), (1, ))
    assert_size_stride(arg829_1, (128, 512), (512, 1))
    assert_size_stride(arg830_1, (128, ), (1, ))
    assert_size_stride(arg831_1, (512, 128), (128, 1))
    assert_size_stride(arg832_1, (512, ), (1, ))
    assert_size_stride(arg833_1, (128, 512), (512, 1))
    assert_size_stride(arg834_1, (128, ), (1, ))
    assert_size_stride(arg835_1, (512, 128), (128, 1))
    assert_size_stride(arg836_1, (512, ), (1, ))
    assert_size_stride(arg837_1, (128, 512), (512, 1))
    assert_size_stride(arg838_1, (128, ), (1, ))
    assert_size_stride(arg839_1, (512, 128), (128, 1))
    assert_size_stride(arg840_1, (512, ), (1, ))
    assert_size_stride(arg841_1, (128, 512), (512, 1))
    assert_size_stride(arg842_1, (128, ), (1, ))
    assert_size_stride(arg843_1, (128, 512), (512, 1))
    assert_size_stride(arg844_1, (128, ), (1, ))
    assert_size_stride(arg845_1, (128, 128), (128, 1))
    assert_size_stride(arg846_1, (128, ), (1, ))
    assert_size_stride(arg847_1, (128, 128), (128, 1))
    assert_size_stride(arg848_1, (128, ), (1, ))
    assert_size_stride(arg849_1, (128, 512), (512, 1))
    assert_size_stride(arg850_1, (128, ), (1, ))
    assert_size_stride(arg851_1, (128, 128), (128, 1))
    assert_size_stride(arg852_1, (128, ), (1, ))
    assert_size_stride(arg853_1, (512, 128), (128, 1))
    assert_size_stride(arg854_1, (512, ), (1, ))
    assert_size_stride(arg855_1, (128, 512), (512, 1))
    assert_size_stride(arg856_1, (128, ), (1, ))
    assert_size_stride(arg857_1, (512, 128), (128, 1))
    assert_size_stride(arg858_1, (512, ), (1, ))
    assert_size_stride(arg859_1, (128, 512), (512, 1))
    assert_size_stride(arg860_1, (128, ), (1, ))
    assert_size_stride(arg861_1, (512, 128), (128, 1))
    assert_size_stride(arg862_1, (512, ), (1, ))
    assert_size_stride(arg863_1, (128, 512), (512, 1))
    assert_size_stride(arg864_1, (128, ), (1, ))
    assert_size_stride(arg865_1, (512, 128), (128, 1))
    assert_size_stride(arg866_1, (512, ), (1, ))
    assert_size_stride(arg867_1, (128, 512), (512, 1))
    assert_size_stride(arg868_1, (128, ), (1, ))
    assert_size_stride(arg869_1, (512, 128), (128, 1))
    assert_size_stride(arg870_1, (512, ), (1, ))
    assert_size_stride(arg871_1, (128, 512), (512, 1))
    assert_size_stride(arg872_1, (128, ), (1, ))
    assert_size_stride(arg873_1, (128, 512), (512, 1))
    assert_size_stride(arg874_1, (128, ), (1, ))
    assert_size_stride(arg875_1, (128, 128), (128, 1))
    assert_size_stride(arg876_1, (128, ), (1, ))
    assert_size_stride(arg877_1, (128, 128), (128, 1))
    assert_size_stride(arg878_1, (128, ), (1, ))
    assert_size_stride(arg879_1, (128, 512), (512, 1))
    assert_size_stride(arg880_1, (128, ), (1, ))
    assert_size_stride(arg881_1, (128, 128), (128, 1))
    assert_size_stride(arg882_1, (128, ), (1, ))
    assert_size_stride(arg883_1, (512, 128), (128, 1))
    assert_size_stride(arg884_1, (512, ), (1, ))
    assert_size_stride(arg885_1, (128, 512), (512, 1))
    assert_size_stride(arg886_1, (128, ), (1, ))
    assert_size_stride(arg887_1, (512, 128), (128, 1))
    assert_size_stride(arg888_1, (512, ), (1, ))
    assert_size_stride(arg889_1, (128, 512), (512, 1))
    assert_size_stride(arg890_1, (128, ), (1, ))
    assert_size_stride(arg891_1, (512, 128), (128, 1))
    assert_size_stride(arg892_1, (512, ), (1, ))
    assert_size_stride(arg893_1, (128, 512), (512, 1))
    assert_size_stride(arg894_1, (128, ), (1, ))
    assert_size_stride(arg895_1, (512, 128), (128, 1))
    assert_size_stride(arg896_1, (512, ), (1, ))
    assert_size_stride(arg897_1, (128, 512), (512, 1))
    assert_size_stride(arg898_1, (128, ), (1, ))
    assert_size_stride(arg899_1, (512, 128), (128, 1))
    assert_size_stride(arg900_1, (512, ), (1, ))
    assert_size_stride(arg901_1, (128, 512), (512, 1))
    assert_size_stride(arg902_1, (128, ), (1, ))
    assert_size_stride(arg903_1, (128, 512), (512, 1))
    assert_size_stride(arg904_1, (128, ), (1, ))
    assert_size_stride(arg905_1, (128, 128), (128, 1))
    assert_size_stride(arg906_1, (128, ), (1, ))
    assert_size_stride(arg907_1, (128, 128), (128, 1))
    assert_size_stride(arg908_1, (128, ), (1, ))
    assert_size_stride(arg909_1, (128, 512), (512, 1))
    assert_size_stride(arg910_1, (128, ), (1, ))
    assert_size_stride(arg911_1, (128, 128), (128, 1))
    assert_size_stride(arg912_1, (128, ), (1, ))
    assert_size_stride(arg913_1, (512, 128), (128, 1))
    assert_size_stride(arg914_1, (512, ), (1, ))
    assert_size_stride(arg915_1, (128, 512), (512, 1))
    assert_size_stride(arg916_1, (128, ), (1, ))
    assert_size_stride(arg917_1, (512, 128), (128, 1))
    assert_size_stride(arg918_1, (512, ), (1, ))
    assert_size_stride(arg919_1, (128, 512), (512, 1))
    assert_size_stride(arg920_1, (128, ), (1, ))
    assert_size_stride(arg921_1, (512, 128), (128, 1))
    assert_size_stride(arg922_1, (512, ), (1, ))
    assert_size_stride(arg923_1, (128, 512), (512, 1))
    assert_size_stride(arg924_1, (128, ), (1, ))
    assert_size_stride(arg925_1, (512, 128), (128, 1))
    assert_size_stride(arg926_1, (512, ), (1, ))
    assert_size_stride(arg927_1, (128, 512), (512, 1))
    assert_size_stride(arg928_1, (128, ), (1, ))
    assert_size_stride(arg929_1, (512, 128), (128, 1))
    assert_size_stride(arg930_1, (512, ), (1, ))
    assert_size_stride(arg931_1, (128, 512), (512, 1))
    assert_size_stride(arg932_1, (128, ), (1, ))
    assert_size_stride(arg933_1, (128, 512), (512, 1))
    assert_size_stride(arg934_1, (128, ), (1, ))
    assert_size_stride(arg935_1, (128, 128), (128, 1))
    assert_size_stride(arg936_1, (128, ), (1, ))
    assert_size_stride(arg937_1, (128, 128), (128, 1))
    assert_size_stride(arg938_1, (128, ), (1, ))
    assert_size_stride(arg939_1, (128, 512), (512, 1))
    assert_size_stride(arg940_1, (128, ), (1, ))
    assert_size_stride(arg941_1, (128, 128), (128, 1))
    assert_size_stride(arg942_1, (128, ), (1, ))
    assert_size_stride(arg943_1, (512, 128), (128, 1))
    assert_size_stride(arg944_1, (512, ), (1, ))
    assert_size_stride(arg945_1, (128, 512), (512, 1))
    assert_size_stride(arg946_1, (128, ), (1, ))
    assert_size_stride(arg947_1, (512, 128), (128, 1))
    assert_size_stride(arg948_1, (512, ), (1, ))
    assert_size_stride(arg949_1, (128, 512), (512, 1))
    assert_size_stride(arg950_1, (128, ), (1, ))
    assert_size_stride(arg951_1, (512, 128), (128, 1))
    assert_size_stride(arg952_1, (512, ), (1, ))
    assert_size_stride(arg953_1, (128, 512), (512, 1))
    assert_size_stride(arg954_1, (128, ), (1, ))
    assert_size_stride(arg955_1, (512, 128), (128, 1))
    assert_size_stride(arg956_1, (512, ), (1, ))
    assert_size_stride(arg957_1, (128, 512), (512, 1))
    assert_size_stride(arg958_1, (128, ), (1, ))
    assert_size_stride(arg959_1, (512, 128), (128, 1))
    assert_size_stride(arg960_1, (512, ), (1, ))
    assert_size_stride(arg961_1, (128, 512), (512, 1))
    assert_size_stride(arg962_1, (128, ), (1, ))
    assert_size_stride(arg963_1, (128, 512), (512, 1))
    assert_size_stride(arg964_1, (128, ), (1, ))
    assert_size_stride(arg965_1, (128, 128), (128, 1))
    assert_size_stride(arg966_1, (128, ), (1, ))
    assert_size_stride(arg967_1, (128, 128), (128, 1))
    assert_size_stride(arg968_1, (128, ), (1, ))
    assert_size_stride(arg969_1, (128, 512), (512, 1))
    assert_size_stride(arg970_1, (128, ), (1, ))
    assert_size_stride(arg971_1, (128, 128), (128, 1))
    assert_size_stride(arg972_1, (128, ), (1, ))
    assert_size_stride(arg973_1, (512, 128), (128, 1))
    assert_size_stride(arg974_1, (512, ), (1, ))
    assert_size_stride(arg975_1, (128, 512), (512, 1))
    assert_size_stride(arg976_1, (128, ), (1, ))
    assert_size_stride(arg977_1, (512, 128), (128, 1))
    assert_size_stride(arg978_1, (512, ), (1, ))
    assert_size_stride(arg979_1, (128, 512), (512, 1))
    assert_size_stride(arg980_1, (128, ), (1, ))
    assert_size_stride(arg981_1, (512, 128), (128, 1))
    assert_size_stride(arg982_1, (512, ), (1, ))
    assert_size_stride(arg983_1, (128, 512), (512, 1))
    assert_size_stride(arg984_1, (128, ), (1, ))
    assert_size_stride(arg985_1, (512, 128), (128, 1))
    assert_size_stride(arg986_1, (512, ), (1, ))
    assert_size_stride(arg987_1, (128, 512), (512, 1))
    assert_size_stride(arg988_1, (128, ), (1, ))
    assert_size_stride(arg989_1, (512, 128), (128, 1))
    assert_size_stride(arg990_1, (512, ), (1, ))
    assert_size_stride(arg991_1, (128, 512), (512, 1))
    assert_size_stride(arg992_1, (128, ), (1, ))
    assert_size_stride(arg993_1, (128, 512), (512, 1))
    assert_size_stride(arg994_1, (128, ), (1, ))
    assert_size_stride(arg995_1, (128, 128), (128, 1))
    assert_size_stride(arg996_1, (128, ), (1, ))
    assert_size_stride(arg997_1, (128, 128), (128, 1))
    assert_size_stride(arg998_1, (128, ), (1, ))
    assert_size_stride(arg999_1, (128, 512), (512, 1))
    assert_size_stride(arg1000_1, (128, ), (1, ))
    assert_size_stride(arg1001_1, (128, 128), (128, 1))
    assert_size_stride(arg1002_1, (128, ), (1, ))
    assert_size_stride(arg1003_1, (512, 128), (128, 1))
    assert_size_stride(arg1004_1, (512, ), (1, ))
    assert_size_stride(arg1005_1, (128, 512), (512, 1))
    assert_size_stride(arg1006_1, (128, ), (1, ))
    assert_size_stride(arg1007_1, (512, 128), (128, 1))
    assert_size_stride(arg1008_1, (512, ), (1, ))
    assert_size_stride(arg1009_1, (128, 512), (512, 1))
    assert_size_stride(arg1010_1, (128, ), (1, ))
    assert_size_stride(arg1011_1, (512, 128), (128, 1))
    assert_size_stride(arg1012_1, (512, ), (1, ))
    assert_size_stride(arg1013_1, (128, 512), (512, 1))
    assert_size_stride(arg1014_1, (128, ), (1, ))
    assert_size_stride(arg1015_1, (512, 128), (128, 1))
    assert_size_stride(arg1016_1, (512, ), (1, ))
    assert_size_stride(arg1017_1, (128, 512), (512, 1))
    assert_size_stride(arg1018_1, (128, ), (1, ))
    assert_size_stride(arg1019_1, (512, 128), (128, 1))
    assert_size_stride(arg1020_1, (512, ), (1, ))
    assert_size_stride(arg1021_1, (128, 512), (512, 1))
    assert_size_stride(arg1022_1, (128, ), (1, ))
    assert_size_stride(arg1023_1, (128, 512), (512, 1))
    assert_size_stride(arg1024_1, (128, ), (1, ))
    assert_size_stride(arg1025_1, (128, 128), (128, 1))
    assert_size_stride(arg1026_1, (128, ), (1, ))
    assert_size_stride(arg1027_1, (128, 128), (128, 1))
    assert_size_stride(arg1028_1, (128, ), (1, ))
    assert_size_stride(arg1029_1, (128, 512), (512, 1))
    assert_size_stride(arg1030_1, (128, ), (1, ))
    assert_size_stride(arg1031_1, (128, 128), (128, 1))
    assert_size_stride(arg1032_1, (128, ), (1, ))
    assert_size_stride(arg1033_1, (512, 128), (128, 1))
    assert_size_stride(arg1034_1, (512, ), (1, ))
    assert_size_stride(arg1035_1, (128, 512), (512, 1))
    assert_size_stride(arg1036_1, (128, ), (1, ))
    assert_size_stride(arg1037_1, (512, 128), (128, 1))
    assert_size_stride(arg1038_1, (512, ), (1, ))
    assert_size_stride(arg1039_1, (128, 512), (512, 1))
    assert_size_stride(arg1040_1, (128, ), (1, ))
    assert_size_stride(arg1041_1, (512, 128), (128, 1))
    assert_size_stride(arg1042_1, (512, ), (1, ))
    assert_size_stride(arg1043_1, (128, 512), (512, 1))
    assert_size_stride(arg1044_1, (128, ), (1, ))
    assert_size_stride(arg1045_1, (512, 128), (128, 1))
    assert_size_stride(arg1046_1, (512, ), (1, ))
    assert_size_stride(arg1047_1, (128, 512), (512, 1))
    assert_size_stride(arg1048_1, (128, ), (1, ))
    assert_size_stride(arg1049_1, (512, 128), (128, 1))
    assert_size_stride(arg1050_1, (512, ), (1, ))
    assert_size_stride(arg1051_1, (128, 512), (512, 1))
    assert_size_stride(arg1052_1, (128, ), (1, ))
    assert_size_stride(arg1053_1, (128, 512), (512, 1))
    assert_size_stride(arg1054_1, (128, ), (1, ))
    assert_size_stride(arg1055_1, (128, 128), (128, 1))
    assert_size_stride(arg1056_1, (128, ), (1, ))
    assert_size_stride(arg1057_1, (128, 128), (128, 1))
    assert_size_stride(arg1058_1, (128, ), (1, ))
    assert_size_stride(arg1059_1, (128, 512), (512, 1))
    assert_size_stride(arg1060_1, (128, ), (1, ))
    assert_size_stride(arg1061_1, (128, 128), (128, 1))
    assert_size_stride(arg1062_1, (128, ), (1, ))
    assert_size_stride(arg1063_1, (512, 128), (128, 1))
    assert_size_stride(arg1064_1, (512, ), (1, ))
    assert_size_stride(arg1065_1, (128, 512), (512, 1))
    assert_size_stride(arg1066_1, (128, ), (1, ))
    assert_size_stride(arg1067_1, (512, 128), (128, 1))
    assert_size_stride(arg1068_1, (512, ), (1, ))
    assert_size_stride(arg1069_1, (128, 512), (512, 1))
    assert_size_stride(arg1070_1, (128, ), (1, ))
    assert_size_stride(arg1071_1, (512, 128), (128, 1))
    assert_size_stride(arg1072_1, (512, ), (1, ))
    assert_size_stride(arg1073_1, (128, 512), (512, 1))
    assert_size_stride(arg1074_1, (128, ), (1, ))
    assert_size_stride(arg1075_1, (512, 128), (128, 1))
    assert_size_stride(arg1076_1, (512, ), (1, ))
    assert_size_stride(arg1077_1, (128, 512), (512, 1))
    assert_size_stride(arg1078_1, (128, ), (1, ))
    assert_size_stride(arg1079_1, (512, 128), (128, 1))
    assert_size_stride(arg1080_1, (512, ), (1, ))
    assert_size_stride(arg1081_1, (128, 512), (512, 1))
    assert_size_stride(arg1082_1, (128, ), (1, ))
    assert_size_stride(arg1083_1, (128, 512), (512, 1))
    assert_size_stride(arg1084_1, (128, ), (1, ))
    assert_size_stride(arg1085_1, (128, 128), (128, 1))
    assert_size_stride(arg1086_1, (128, ), (1, ))
    assert_size_stride(arg1087_1, (128, 128), (128, 1))
    assert_size_stride(arg1088_1, (128, ), (1, ))
    assert_size_stride(arg1089_1, (128, 512), (512, 1))
    assert_size_stride(arg1090_1, (128, ), (1, ))
    assert_size_stride(arg1091_1, (128, 128), (128, 1))
    assert_size_stride(arg1092_1, (128, ), (1, ))
    assert_size_stride(arg1093_1, (512, 128), (128, 1))
    assert_size_stride(arg1094_1, (512, ), (1, ))
    assert_size_stride(arg1095_1, (128, 512), (512, 1))
    assert_size_stride(arg1096_1, (128, ), (1, ))
    assert_size_stride(arg1097_1, (512, 128), (128, 1))
    assert_size_stride(arg1098_1, (512, ), (1, ))
    assert_size_stride(arg1099_1, (128, 512), (512, 1))
    assert_size_stride(arg1100_1, (128, ), (1, ))
    assert_size_stride(arg1101_1, (512, 128), (128, 1))
    assert_size_stride(arg1102_1, (512, ), (1, ))
    assert_size_stride(arg1103_1, (128, 512), (512, 1))
    assert_size_stride(arg1104_1, (128, ), (1, ))
    assert_size_stride(arg1105_1, (512, 128), (128, 1))
    assert_size_stride(arg1106_1, (512, ), (1, ))
    assert_size_stride(arg1107_1, (128, 512), (512, 1))
    assert_size_stride(arg1108_1, (128, ), (1, ))
    assert_size_stride(arg1109_1, (512, 128), (128, 1))
    assert_size_stride(arg1110_1, (512, ), (1, ))
    assert_size_stride(arg1111_1, (2, 512), (512, 1))
    assert_size_stride(arg1112_1, (2, ), (1, ))
    assert_size_stride(arg1113_1, (1, 512), (512, 1))
    assert_size_stride(arg1114_1, (1, 128), (128, 1))
    assert_size_stride(arg1115_1, (1, ), (1, ))
    assert_size_stride(arg1116_1, (1, ), (1, ))
    buf0 = empty((1, 128, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_0(c_void_p(arg1114_1.data_ptr()), c_void_p(arg386_1.data_ptr()), c_void_p(buf0.data_ptr()))
    del arg1114_1
    del arg386_1
    buf1 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [inputs_embeds_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg388_1, reinterpret_tensor(buf0, (128, 384), (384, 1), 0), reinterpret_tensor(arg387_1, (384, 512), (1, 384), 0), alpha=1, beta=1, out=buf1)
    del arg387_1
    del arg388_1
    del buf0
    buf2 = reinterpret_tensor(buf1, (1, 128, 512), (65536, 512, 1), 0); del buf1  # reuse
    cpp_fused_add_embedding_mul_zeros_1(c_void_p(buf2.data_ptr()), c_void_p(arg1113_1.data_ptr()), c_void_p(arg389_1.data_ptr()), c_void_p(arg390_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg1_1.data_ptr()))
    del arg0_1
    del arg1113_1
    del arg1_1
    del arg389_1
    del arg390_1
    buf3 = empty((128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [layer_input_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg394_1, reinterpret_tensor(buf2, (128, 512), (512, 1), 0), reinterpret_tensor(arg393_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf3)
    del arg393_1
    del arg394_1
    buf4 = reinterpret_tensor(buf3, (1, 128, 128), (16384, 128, 1), 0); del buf3  # reuse
    cpp_fused_add_mul_2(c_void_p(buf4.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(arg5_1.data_ptr()))
    del arg4_1
    del arg5_1
    buf5 = empty((128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [mixed_query_layer], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg396_1, reinterpret_tensor(buf4, (128, 128), (128, 1), 0), reinterpret_tensor(arg395_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf5)
    del arg395_1
    del arg396_1
    buf6 = empty((128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [mixed_key_layer], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg398_1, reinterpret_tensor(buf4, (128, 128), (128, 1), 0), reinterpret_tensor(arg397_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf6)
    del arg397_1
    del arg398_1
    buf7 = reinterpret_tensor(buf4, (128, 128), (128, 1), 0); del buf4  # reuse
    # Source Nodes: [mixed_value_layer], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg400_1, reinterpret_tensor(buf2, (128, 512), (512, 1), 0), reinterpret_tensor(arg399_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf7)
    del arg399_1
    del arg400_1
    buf8 = reinterpret_tensor(buf5, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf5  # reuse
    buf9 = reinterpret_tensor(buf6, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf6  # reuse
    buf10 = reinterpret_tensor(buf7, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf7  # reuse
    cpp_fused_3(c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf11 = aten._scaled_dot_product_flash_attention(buf8, buf9, buf10, scale=0.17677669529663687)
    del buf10
    buf12 = buf11[0]
    del buf11
    buf19 = reinterpret_tensor(buf9, (128, 128), (128, 1), 0); del buf9  # reuse
    # Source Nodes: [layer_outputs], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg402_1, reinterpret_tensor(buf12, (128, 128), (128, 1), 0), reinterpret_tensor(arg401_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf19)
    del arg401_1
    del arg402_1
    buf20 = reinterpret_tensor(buf12, (128, 128), (128, 1), 0); del buf12  # reuse
    # Source Nodes: [layer_input], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg392_1, reinterpret_tensor(buf2, (128, 512), (512, 1), 0), reinterpret_tensor(arg391_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf20)
    del arg391_1
    del arg392_1
    buf21 = reinterpret_tensor(buf19, (1, 128, 128), (16384, 128, 1), 0); del buf19  # reuse
    cpp_fused_add_mul_4(c_void_p(buf21.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(arg7_1.data_ptr()))
    del arg2_1
    del arg3_1
    del arg6_1
    del arg7_1
    buf22 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg404_1, reinterpret_tensor(buf21, (128, 128), (128, 1), 0), reinterpret_tensor(arg403_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf22)
    del arg403_1
    del arg404_1
    buf23 = reinterpret_tensor(buf22, (1, 128, 512), (65536, 512, 1), 0); del buf22  # reuse
    cpp_fused_relu_5(c_void_p(buf23.data_ptr()))
    buf24 = buf20; del buf20  # reuse
    # Source Nodes: [layer_outputs_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg406_1, reinterpret_tensor(buf23, (128, 512), (512, 1), 0), reinterpret_tensor(arg405_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf24)
    del arg405_1
    del arg406_1
    buf25 = buf21; del buf21  # reuse
    cpp_fused_add_mul_6(c_void_p(buf25.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(arg9_1.data_ptr()))
    del arg8_1
    del arg9_1
    buf26 = reinterpret_tensor(buf23, (128, 512), (512, 1), 0); del buf23  # reuse
    # Source Nodes: [hidden_states_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg408_1, reinterpret_tensor(buf25, (128, 128), (128, 1), 0), reinterpret_tensor(arg407_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf26)
    del arg407_1
    del arg408_1
    buf27 = reinterpret_tensor(buf26, (1, 128, 512), (65536, 512, 1), 0); del buf26  # reuse
    cpp_fused_relu_7(c_void_p(buf27.data_ptr()))
    buf28 = buf24; del buf24  # reuse
    # Source Nodes: [layer_outputs_5], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg410_1, reinterpret_tensor(buf27, (128, 512), (512, 1), 0), reinterpret_tensor(arg409_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf28)
    del arg409_1
    del arg410_1
    buf29 = buf25; del buf25  # reuse
    cpp_fused_add_mul_8(c_void_p(buf29.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(arg11_1.data_ptr()))
    del arg10_1
    del arg11_1
    buf30 = reinterpret_tensor(buf27, (128, 512), (512, 1), 0); del buf27  # reuse
    # Source Nodes: [hidden_states_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg412_1, reinterpret_tensor(buf29, (128, 128), (128, 1), 0), reinterpret_tensor(arg411_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf30)
    del arg411_1
    del arg412_1
    buf31 = reinterpret_tensor(buf30, (1, 128, 512), (65536, 512, 1), 0); del buf30  # reuse
    cpp_fused_relu_9(c_void_p(buf31.data_ptr()))
    buf32 = buf28; del buf28  # reuse
    # Source Nodes: [layer_outputs_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg414_1, reinterpret_tensor(buf31, (128, 512), (512, 1), 0), reinterpret_tensor(arg413_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf32)
    del arg413_1
    del arg414_1
    buf33 = buf29; del buf29  # reuse
    cpp_fused_add_mul_10(c_void_p(buf33.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(arg13_1.data_ptr()))
    del arg12_1
    del arg13_1
    buf34 = reinterpret_tensor(buf31, (128, 512), (512, 1), 0); del buf31  # reuse
    # Source Nodes: [hidden_states_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg416_1, reinterpret_tensor(buf33, (128, 128), (128, 1), 0), reinterpret_tensor(arg415_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf34)
    del arg415_1
    del arg416_1
    buf35 = reinterpret_tensor(buf34, (1, 128, 512), (65536, 512, 1), 0); del buf34  # reuse
    cpp_fused_relu_11(c_void_p(buf35.data_ptr()))
    buf36 = buf32; del buf32  # reuse
    # Source Nodes: [layer_output], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg418_1, reinterpret_tensor(buf35, (128, 512), (512, 1), 0), reinterpret_tensor(arg417_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf36)
    del arg417_1
    del arg418_1
    buf37 = buf33; del buf33  # reuse
    cpp_fused_add_mul_12(c_void_p(buf37.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(arg15_1.data_ptr()))
    del arg14_1
    del arg15_1
    buf38 = reinterpret_tensor(buf35, (128, 512), (512, 1), 0); del buf35  # reuse
    # Source Nodes: [layer_outputs_11], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg420_1, reinterpret_tensor(buf37, (128, 128), (128, 1), 0), reinterpret_tensor(arg419_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf38)
    del arg419_1
    del arg420_1
    buf39 = buf2; del buf2  # reuse
    cpp_fused_add_mul_13(c_void_p(buf39.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(arg17_1.data_ptr()))
    del arg16_1
    del arg17_1
    buf40 = reinterpret_tensor(buf37, (128, 128), (128, 1), 0); del buf37  # reuse
    # Source Nodes: [layer_input_7], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg424_1, reinterpret_tensor(buf39, (128, 512), (512, 1), 0), reinterpret_tensor(arg423_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf40)
    del arg423_1
    del arg424_1
    buf41 = reinterpret_tensor(buf40, (1, 128, 128), (16384, 128, 1), 0); del buf40  # reuse
    cpp_fused_add_mul_14(c_void_p(buf41.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(arg21_1.data_ptr()))
    del arg20_1
    del arg21_1
    buf42 = buf36; del buf36  # reuse
    # Source Nodes: [mixed_query_layer_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg426_1, reinterpret_tensor(buf41, (128, 128), (128, 1), 0), reinterpret_tensor(arg425_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf42)
    del arg425_1
    del arg426_1
    buf43 = reinterpret_tensor(buf8, (128, 128), (128, 1), 0); del buf8  # reuse
    # Source Nodes: [mixed_key_layer_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg428_1, reinterpret_tensor(buf41, (128, 128), (128, 1), 0), reinterpret_tensor(arg427_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf43)
    del arg427_1
    del arg428_1
    buf44 = reinterpret_tensor(buf41, (128, 128), (128, 1), 0); del buf41  # reuse
    # Source Nodes: [mixed_value_layer_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg430_1, reinterpret_tensor(buf39, (128, 512), (512, 1), 0), reinterpret_tensor(arg429_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf44)
    del arg429_1
    del arg430_1
    buf45 = reinterpret_tensor(buf42, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf42  # reuse
    buf46 = reinterpret_tensor(buf43, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf43  # reuse
    buf47 = reinterpret_tensor(buf44, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf44  # reuse
    cpp_fused_15(c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf48 = aten._scaled_dot_product_flash_attention(buf45, buf46, buf47, scale=0.17677669529663687)
    del buf45
    buf49 = buf48[0]
    del buf48
    buf56 = reinterpret_tensor(buf47, (128, 128), (128, 1), 0); del buf47  # reuse
    # Source Nodes: [layer_outputs_14], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg432_1, reinterpret_tensor(buf49, (128, 128), (128, 1), 0), reinterpret_tensor(arg431_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf56)
    del arg431_1
    del arg432_1
    buf57 = reinterpret_tensor(buf49, (128, 128), (128, 1), 0); del buf49  # reuse
    # Source Nodes: [layer_input_5], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg422_1, reinterpret_tensor(buf39, (128, 512), (512, 1), 0), reinterpret_tensor(arg421_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf57)
    del arg421_1
    del arg422_1
    buf58 = reinterpret_tensor(buf56, (1, 128, 128), (16384, 128, 1), 0); del buf56  # reuse
    cpp_fused_add_mul_16(c_void_p(buf58.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(arg23_1.data_ptr()))
    del arg18_1
    del arg19_1
    del arg22_1
    del arg23_1
    buf59 = buf38; del buf38  # reuse
    # Source Nodes: [hidden_states_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg434_1, reinterpret_tensor(buf58, (128, 128), (128, 1), 0), reinterpret_tensor(arg433_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf59)
    del arg433_1
    del arg434_1
    buf60 = reinterpret_tensor(buf59, (1, 128, 512), (65536, 512, 1), 0); del buf59  # reuse
    cpp_fused_relu_17(c_void_p(buf60.data_ptr()))
    buf61 = buf57; del buf57  # reuse
    # Source Nodes: [layer_outputs_16], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg436_1, reinterpret_tensor(buf60, (128, 512), (512, 1), 0), reinterpret_tensor(arg435_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf61)
    del arg435_1
    del arg436_1
    buf62 = buf58; del buf58  # reuse
    cpp_fused_add_mul_18(c_void_p(buf62.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(arg24_1.data_ptr()), c_void_p(arg25_1.data_ptr()))
    del arg24_1
    del arg25_1
    buf63 = reinterpret_tensor(buf60, (128, 512), (512, 1), 0); del buf60  # reuse
    # Source Nodes: [hidden_states_11], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg438_1, reinterpret_tensor(buf62, (128, 128), (128, 1), 0), reinterpret_tensor(arg437_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf63)
    del arg437_1
    del arg438_1
    buf64 = reinterpret_tensor(buf63, (1, 128, 512), (65536, 512, 1), 0); del buf63  # reuse
    cpp_fused_relu_19(c_void_p(buf64.data_ptr()))
    buf65 = buf61; del buf61  # reuse
    # Source Nodes: [layer_outputs_19], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg440_1, reinterpret_tensor(buf64, (128, 512), (512, 1), 0), reinterpret_tensor(arg439_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf65)
    del arg439_1
    del arg440_1
    buf66 = buf62; del buf62  # reuse
    cpp_fused_add_mul_20(c_void_p(buf66.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(arg27_1.data_ptr()))
    del arg26_1
    del arg27_1
    buf67 = reinterpret_tensor(buf64, (128, 512), (512, 1), 0); del buf64  # reuse
    # Source Nodes: [hidden_states_13], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg442_1, reinterpret_tensor(buf66, (128, 128), (128, 1), 0), reinterpret_tensor(arg441_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf67)
    del arg441_1
    del arg442_1
    buf68 = reinterpret_tensor(buf67, (1, 128, 512), (65536, 512, 1), 0); del buf67  # reuse
    cpp_fused_relu_21(c_void_p(buf68.data_ptr()))
    buf69 = buf65; del buf65  # reuse
    # Source Nodes: [layer_outputs_22], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg444_1, reinterpret_tensor(buf68, (128, 512), (512, 1), 0), reinterpret_tensor(arg443_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf69)
    del arg443_1
    del arg444_1
    buf70 = buf66; del buf66  # reuse
    cpp_fused_add_mul_22(c_void_p(buf70.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(arg29_1.data_ptr()))
    del arg28_1
    del arg29_1
    buf71 = reinterpret_tensor(buf68, (128, 512), (512, 1), 0); del buf68  # reuse
    # Source Nodes: [hidden_states_15], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg446_1, reinterpret_tensor(buf70, (128, 128), (128, 1), 0), reinterpret_tensor(arg445_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf71)
    del arg445_1
    del arg446_1
    buf72 = reinterpret_tensor(buf71, (1, 128, 512), (65536, 512, 1), 0); del buf71  # reuse
    cpp_fused_relu_23(c_void_p(buf72.data_ptr()))
    buf73 = buf69; del buf69  # reuse
    # Source Nodes: [layer_output_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg448_1, reinterpret_tensor(buf72, (128, 512), (512, 1), 0), reinterpret_tensor(arg447_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf73)
    del arg447_1
    del arg448_1
    buf74 = buf70; del buf70  # reuse
    cpp_fused_add_mul_24(c_void_p(buf74.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(arg31_1.data_ptr()))
    del arg30_1
    del arg31_1
    buf75 = reinterpret_tensor(buf72, (128, 512), (512, 1), 0); del buf72  # reuse
    # Source Nodes: [layer_outputs_25], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg450_1, reinterpret_tensor(buf74, (128, 128), (128, 1), 0), reinterpret_tensor(arg449_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf75)
    del arg449_1
    del arg450_1
    buf76 = buf39; del buf39  # reuse
    cpp_fused_add_mul_25(c_void_p(buf76.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(arg33_1.data_ptr()))
    del arg32_1
    del arg33_1
    buf77 = reinterpret_tensor(buf74, (128, 128), (128, 1), 0); del buf74  # reuse
    # Source Nodes: [layer_input_12], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg454_1, reinterpret_tensor(buf76, (128, 512), (512, 1), 0), reinterpret_tensor(arg453_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf77)
    del arg453_1
    del arg454_1
    buf78 = reinterpret_tensor(buf77, (1, 128, 128), (16384, 128, 1), 0); del buf77  # reuse
    cpp_fused_add_mul_26(c_void_p(buf78.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(arg37_1.data_ptr()))
    del arg36_1
    del arg37_1
    buf79 = buf73; del buf73  # reuse
    # Source Nodes: [mixed_query_layer_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg456_1, reinterpret_tensor(buf78, (128, 128), (128, 1), 0), reinterpret_tensor(arg455_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf79)
    del arg455_1
    del arg456_1
    buf80 = reinterpret_tensor(buf46, (128, 128), (128, 1), 0); del buf46  # reuse
    # Source Nodes: [mixed_key_layer_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg458_1, reinterpret_tensor(buf78, (128, 128), (128, 1), 0), reinterpret_tensor(arg457_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf80)
    del arg457_1
    del arg458_1
    buf81 = reinterpret_tensor(buf78, (128, 128), (128, 1), 0); del buf78  # reuse
    # Source Nodes: [mixed_value_layer_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg460_1, reinterpret_tensor(buf76, (128, 512), (512, 1), 0), reinterpret_tensor(arg459_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf81)
    del arg459_1
    del arg460_1
    buf82 = reinterpret_tensor(buf79, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf79  # reuse
    buf83 = reinterpret_tensor(buf80, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf80  # reuse
    buf84 = reinterpret_tensor(buf81, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf81  # reuse
    cpp_fused_27(c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf85 = aten._scaled_dot_product_flash_attention(buf82, buf83, buf84, scale=0.17677669529663687)
    del buf82
    buf86 = buf85[0]
    del buf85
    buf93 = reinterpret_tensor(buf84, (128, 128), (128, 1), 0); del buf84  # reuse
    # Source Nodes: [layer_outputs_28], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg462_1, reinterpret_tensor(buf86, (128, 128), (128, 1), 0), reinterpret_tensor(arg461_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf93)
    del arg461_1
    del arg462_1
    buf94 = reinterpret_tensor(buf86, (128, 128), (128, 1), 0); del buf86  # reuse
    # Source Nodes: [layer_input_10], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg452_1, reinterpret_tensor(buf76, (128, 512), (512, 1), 0), reinterpret_tensor(arg451_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf94)
    del arg451_1
    del arg452_1
    buf95 = reinterpret_tensor(buf93, (1, 128, 128), (16384, 128, 1), 0); del buf93  # reuse
    cpp_fused_add_mul_28(c_void_p(buf95.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(arg35_1.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(arg39_1.data_ptr()))
    del arg34_1
    del arg35_1
    del arg38_1
    del arg39_1
    buf96 = buf75; del buf75  # reuse
    # Source Nodes: [hidden_states_18], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg464_1, reinterpret_tensor(buf95, (128, 128), (128, 1), 0), reinterpret_tensor(arg463_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf96)
    del arg463_1
    del arg464_1
    buf97 = reinterpret_tensor(buf96, (1, 128, 512), (65536, 512, 1), 0); del buf96  # reuse
    cpp_fused_relu_29(c_void_p(buf97.data_ptr()))
    buf98 = buf94; del buf94  # reuse
    # Source Nodes: [layer_outputs_30], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg466_1, reinterpret_tensor(buf97, (128, 512), (512, 1), 0), reinterpret_tensor(arg465_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf98)
    del arg465_1
    del arg466_1
    buf99 = buf95; del buf95  # reuse
    cpp_fused_add_mul_30(c_void_p(buf99.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(arg41_1.data_ptr()))
    del arg40_1
    del arg41_1
    buf100 = reinterpret_tensor(buf97, (128, 512), (512, 1), 0); del buf97  # reuse
    # Source Nodes: [hidden_states_20], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg468_1, reinterpret_tensor(buf99, (128, 128), (128, 1), 0), reinterpret_tensor(arg467_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf100)
    del arg467_1
    del arg468_1
    buf101 = reinterpret_tensor(buf100, (1, 128, 512), (65536, 512, 1), 0); del buf100  # reuse
    cpp_fused_relu_31(c_void_p(buf101.data_ptr()))
    buf102 = buf98; del buf98  # reuse
    # Source Nodes: [layer_outputs_33], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg470_1, reinterpret_tensor(buf101, (128, 512), (512, 1), 0), reinterpret_tensor(arg469_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf102)
    del arg469_1
    del arg470_1
    buf103 = reinterpret_tensor(buf102, (1, 128, 128), (16384, 128, 1), 0); del buf102  # reuse
    cpp_fused_add_mul_32(c_void_p(buf103.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(arg42_1.data_ptr()), c_void_p(arg43_1.data_ptr()))
    del arg42_1
    del arg43_1
    buf104 = reinterpret_tensor(buf101, (128, 512), (512, 1), 0); del buf101  # reuse
    # Source Nodes: [hidden_states_22], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg472_1, reinterpret_tensor(buf103, (128, 128), (128, 1), 0), reinterpret_tensor(arg471_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf104)
    del arg471_1
    del arg472_1
    buf105 = reinterpret_tensor(buf104, (1, 128, 512), (65536, 512, 1), 0); del buf104  # reuse
    cpp_fused_relu_33(c_void_p(buf105.data_ptr()))
    buf106 = reinterpret_tensor(buf99, (128, 128), (128, 1), 0); del buf99  # reuse
    # Source Nodes: [layer_outputs_36], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg474_1, reinterpret_tensor(buf105, (128, 512), (512, 1), 0), reinterpret_tensor(arg473_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf106)
    del arg473_1
    del arg474_1
    buf107 = buf103; del buf103  # reuse
    cpp_fused_add_mul_34(c_void_p(buf107.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(arg44_1.data_ptr()), c_void_p(arg45_1.data_ptr()))
    del arg44_1
    del arg45_1
    buf108 = reinterpret_tensor(buf105, (128, 512), (512, 1), 0); del buf105  # reuse
    # Source Nodes: [hidden_states_24], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg476_1, reinterpret_tensor(buf107, (128, 128), (128, 1), 0), reinterpret_tensor(arg475_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf108)
    del arg475_1
    del arg476_1
    buf109 = reinterpret_tensor(buf108, (1, 128, 512), (65536, 512, 1), 0); del buf108  # reuse
    cpp_fused_relu_35(c_void_p(buf109.data_ptr()))
    buf110 = buf106; del buf106  # reuse
    # Source Nodes: [layer_output_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg478_1, reinterpret_tensor(buf109, (128, 512), (512, 1), 0), reinterpret_tensor(arg477_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf110)
    del arg477_1
    del arg478_1
    buf111 = buf107; del buf107  # reuse
    cpp_fused_add_mul_36(c_void_p(buf111.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(arg47_1.data_ptr()))
    del arg46_1
    del arg47_1
    buf112 = reinterpret_tensor(buf109, (128, 512), (512, 1), 0); del buf109  # reuse
    # Source Nodes: [layer_outputs_39], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg480_1, reinterpret_tensor(buf111, (128, 128), (128, 1), 0), reinterpret_tensor(arg479_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf112)
    del arg479_1
    del arg480_1
    buf113 = reinterpret_tensor(buf112, (1, 128, 512), (65536, 512, 1), 0); del buf112  # reuse
    cpp_fused_add_mul_37(c_void_p(buf113.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(arg49_1.data_ptr()))
    del arg48_1
    del arg49_1
    buf114 = reinterpret_tensor(buf111, (128, 128), (128, 1), 0); del buf111  # reuse
    # Source Nodes: [layer_input_17], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg484_1, reinterpret_tensor(buf113, (128, 512), (512, 1), 0), reinterpret_tensor(arg483_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf114)
    del arg483_1
    del arg484_1
    buf115 = reinterpret_tensor(buf114, (1, 128, 128), (16384, 128, 1), 0); del buf114  # reuse
    cpp_fused_add_mul_38(c_void_p(buf115.data_ptr()), c_void_p(arg52_1.data_ptr()), c_void_p(arg53_1.data_ptr()))
    del arg52_1
    del arg53_1
    buf116 = buf110; del buf110  # reuse
    # Source Nodes: [mixed_query_layer_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg486_1, reinterpret_tensor(buf115, (128, 128), (128, 1), 0), reinterpret_tensor(arg485_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf116)
    del arg485_1
    del arg486_1
    buf117 = reinterpret_tensor(buf83, (128, 128), (128, 1), 0); del buf83  # reuse
    # Source Nodes: [mixed_key_layer_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg488_1, reinterpret_tensor(buf115, (128, 128), (128, 1), 0), reinterpret_tensor(arg487_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf117)
    del arg487_1
    del arg488_1
    buf118 = reinterpret_tensor(buf115, (128, 128), (128, 1), 0); del buf115  # reuse
    # Source Nodes: [mixed_value_layer_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg490_1, reinterpret_tensor(buf113, (128, 512), (512, 1), 0), reinterpret_tensor(arg489_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf118)
    del arg489_1
    del arg490_1
    buf119 = reinterpret_tensor(buf116, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf116  # reuse
    buf120 = reinterpret_tensor(buf117, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf117  # reuse
    buf121 = reinterpret_tensor(buf118, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf118  # reuse
    cpp_fused_39(c_void_p(buf119.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf121.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf122 = aten._scaled_dot_product_flash_attention(buf119, buf120, buf121, scale=0.17677669529663687)
    del buf119
    buf123 = buf122[0]
    del buf122
    buf130 = reinterpret_tensor(buf121, (128, 128), (128, 1), 0); del buf121  # reuse
    # Source Nodes: [layer_outputs_42], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg492_1, reinterpret_tensor(buf123, (128, 128), (128, 1), 0), reinterpret_tensor(arg491_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf130)
    del arg491_1
    del arg492_1
    buf131 = reinterpret_tensor(buf123, (128, 128), (128, 1), 0); del buf123  # reuse
    # Source Nodes: [layer_input_15], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg482_1, reinterpret_tensor(buf113, (128, 512), (512, 1), 0), reinterpret_tensor(arg481_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf131)
    del arg481_1
    del arg482_1
    buf132 = reinterpret_tensor(buf130, (1, 128, 128), (16384, 128, 1), 0); del buf130  # reuse
    cpp_fused_add_mul_40(c_void_p(buf132.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(arg50_1.data_ptr()), c_void_p(arg51_1.data_ptr()), c_void_p(arg54_1.data_ptr()), c_void_p(arg55_1.data_ptr()))
    del arg50_1
    del arg51_1
    del arg54_1
    del arg55_1
    buf133 = reinterpret_tensor(buf76, (128, 512), (512, 1), 0); del buf76  # reuse
    # Source Nodes: [hidden_states_27], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg494_1, reinterpret_tensor(buf132, (128, 128), (128, 1), 0), reinterpret_tensor(arg493_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf133)
    del arg493_1
    del arg494_1
    buf134 = reinterpret_tensor(buf133, (1, 128, 512), (65536, 512, 1), 0); del buf133  # reuse
    cpp_fused_relu_41(c_void_p(buf134.data_ptr()))
    buf135 = buf131; del buf131  # reuse
    # Source Nodes: [layer_outputs_44], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg496_1, reinterpret_tensor(buf134, (128, 512), (512, 1), 0), reinterpret_tensor(arg495_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf135)
    del arg495_1
    del arg496_1
    buf136 = buf132; del buf132  # reuse
    cpp_fused_add_mul_42(c_void_p(buf136.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(arg56_1.data_ptr()), c_void_p(arg57_1.data_ptr()))
    del arg56_1
    del arg57_1
    buf137 = reinterpret_tensor(buf134, (128, 512), (512, 1), 0); del buf134  # reuse
    # Source Nodes: [hidden_states_29], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg498_1, reinterpret_tensor(buf136, (128, 128), (128, 1), 0), reinterpret_tensor(arg497_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf137)
    del arg497_1
    del arg498_1
    buf138 = reinterpret_tensor(buf137, (1, 128, 512), (65536, 512, 1), 0); del buf137  # reuse
    cpp_fused_relu_43(c_void_p(buf138.data_ptr()))
    buf139 = buf135; del buf135  # reuse
    # Source Nodes: [layer_outputs_47], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg500_1, reinterpret_tensor(buf138, (128, 512), (512, 1), 0), reinterpret_tensor(arg499_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf139)
    del arg499_1
    del arg500_1
    buf140 = buf136; del buf136  # reuse
    cpp_fused_add_mul_44(c_void_p(buf140.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(arg58_1.data_ptr()), c_void_p(arg59_1.data_ptr()))
    del arg58_1
    del arg59_1
    buf141 = reinterpret_tensor(buf138, (128, 512), (512, 1), 0); del buf138  # reuse
    # Source Nodes: [hidden_states_31], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg502_1, reinterpret_tensor(buf140, (128, 128), (128, 1), 0), reinterpret_tensor(arg501_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf141)
    del arg501_1
    del arg502_1
    buf142 = reinterpret_tensor(buf141, (1, 128, 512), (65536, 512, 1), 0); del buf141  # reuse
    cpp_fused_relu_45(c_void_p(buf142.data_ptr()))
    buf143 = buf139; del buf139  # reuse
    # Source Nodes: [layer_outputs_50], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg504_1, reinterpret_tensor(buf142, (128, 512), (512, 1), 0), reinterpret_tensor(arg503_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf143)
    del arg503_1
    del arg504_1
    buf144 = buf140; del buf140  # reuse
    cpp_fused_add_mul_46(c_void_p(buf144.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(arg60_1.data_ptr()), c_void_p(arg61_1.data_ptr()))
    del arg60_1
    del arg61_1
    buf145 = reinterpret_tensor(buf142, (128, 512), (512, 1), 0); del buf142  # reuse
    # Source Nodes: [hidden_states_33], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg506_1, reinterpret_tensor(buf144, (128, 128), (128, 1), 0), reinterpret_tensor(arg505_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf145)
    del arg505_1
    del arg506_1
    buf146 = reinterpret_tensor(buf145, (1, 128, 512), (65536, 512, 1), 0); del buf145  # reuse
    cpp_fused_relu_47(c_void_p(buf146.data_ptr()))
    buf147 = buf143; del buf143  # reuse
    # Source Nodes: [layer_output_12], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg508_1, reinterpret_tensor(buf146, (128, 512), (512, 1), 0), reinterpret_tensor(arg507_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf147)
    del arg507_1
    del arg508_1
    buf148 = buf144; del buf144  # reuse
    cpp_fused_add_mul_48(c_void_p(buf148.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(arg62_1.data_ptr()), c_void_p(arg63_1.data_ptr()))
    del arg62_1
    del arg63_1
    buf149 = reinterpret_tensor(buf146, (128, 512), (512, 1), 0); del buf146  # reuse
    # Source Nodes: [layer_outputs_53], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg510_1, reinterpret_tensor(buf148, (128, 128), (128, 1), 0), reinterpret_tensor(arg509_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf149)
    del arg509_1
    del arg510_1
    buf150 = buf113; del buf113  # reuse
    cpp_fused_add_mul_49(c_void_p(buf150.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(arg64_1.data_ptr()), c_void_p(arg65_1.data_ptr()))
    del arg64_1
    del arg65_1
    buf151 = reinterpret_tensor(buf148, (128, 128), (128, 1), 0); del buf148  # reuse
    # Source Nodes: [layer_input_22], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg514_1, reinterpret_tensor(buf150, (128, 512), (512, 1), 0), reinterpret_tensor(arg513_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf151)
    del arg513_1
    del arg514_1
    buf152 = reinterpret_tensor(buf151, (1, 128, 128), (16384, 128, 1), 0); del buf151  # reuse
    cpp_fused_add_mul_50(c_void_p(buf152.data_ptr()), c_void_p(arg68_1.data_ptr()), c_void_p(arg69_1.data_ptr()))
    del arg68_1
    del arg69_1
    buf153 = buf147; del buf147  # reuse
    # Source Nodes: [mixed_query_layer_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg516_1, reinterpret_tensor(buf152, (128, 128), (128, 1), 0), reinterpret_tensor(arg515_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf153)
    del arg515_1
    del arg516_1
    buf154 = reinterpret_tensor(buf120, (128, 128), (128, 1), 0); del buf120  # reuse
    # Source Nodes: [mixed_key_layer_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg518_1, reinterpret_tensor(buf152, (128, 128), (128, 1), 0), reinterpret_tensor(arg517_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf154)
    del arg517_1
    del arg518_1
    buf155 = reinterpret_tensor(buf152, (128, 128), (128, 1), 0); del buf152  # reuse
    # Source Nodes: [mixed_value_layer_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg520_1, reinterpret_tensor(buf150, (128, 512), (512, 1), 0), reinterpret_tensor(arg519_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf155)
    del arg519_1
    del arg520_1
    buf156 = reinterpret_tensor(buf153, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf153  # reuse
    buf157 = reinterpret_tensor(buf154, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf154  # reuse
    buf158 = reinterpret_tensor(buf155, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf155  # reuse
    cpp_fused_51(c_void_p(buf156.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf158.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf159 = aten._scaled_dot_product_flash_attention(buf156, buf157, buf158, scale=0.17677669529663687)
    del buf156
    buf160 = buf159[0]
    del buf159
    buf167 = reinterpret_tensor(buf158, (128, 128), (128, 1), 0); del buf158  # reuse
    # Source Nodes: [layer_outputs_56], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg522_1, reinterpret_tensor(buf160, (128, 128), (128, 1), 0), reinterpret_tensor(arg521_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf167)
    del arg521_1
    del arg522_1
    buf168 = reinterpret_tensor(buf160, (128, 128), (128, 1), 0); del buf160  # reuse
    # Source Nodes: [layer_input_20], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg512_1, reinterpret_tensor(buf150, (128, 512), (512, 1), 0), reinterpret_tensor(arg511_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf168)
    del arg511_1
    del arg512_1
    buf169 = reinterpret_tensor(buf167, (1, 128, 128), (16384, 128, 1), 0); del buf167  # reuse
    cpp_fused_add_mul_52(c_void_p(buf169.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(arg66_1.data_ptr()), c_void_p(arg67_1.data_ptr()), c_void_p(arg70_1.data_ptr()), c_void_p(arg71_1.data_ptr()))
    del arg66_1
    del arg67_1
    del arg70_1
    del arg71_1
    buf170 = buf149; del buf149  # reuse
    # Source Nodes: [hidden_states_36], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg524_1, reinterpret_tensor(buf169, (128, 128), (128, 1), 0), reinterpret_tensor(arg523_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf170)
    del arg523_1
    del arg524_1
    buf171 = reinterpret_tensor(buf170, (1, 128, 512), (65536, 512, 1), 0); del buf170  # reuse
    cpp_fused_relu_53(c_void_p(buf171.data_ptr()))
    buf172 = buf168; del buf168  # reuse
    # Source Nodes: [layer_outputs_58], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg526_1, reinterpret_tensor(buf171, (128, 512), (512, 1), 0), reinterpret_tensor(arg525_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf172)
    del arg525_1
    del arg526_1
    buf173 = buf169; del buf169  # reuse
    cpp_fused_add_mul_54(c_void_p(buf173.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(arg72_1.data_ptr()), c_void_p(arg73_1.data_ptr()))
    del arg72_1
    del arg73_1
    buf174 = reinterpret_tensor(buf171, (128, 512), (512, 1), 0); del buf171  # reuse
    # Source Nodes: [hidden_states_38], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg528_1, reinterpret_tensor(buf173, (128, 128), (128, 1), 0), reinterpret_tensor(arg527_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf174)
    del arg527_1
    del arg528_1
    buf175 = reinterpret_tensor(buf174, (1, 128, 512), (65536, 512, 1), 0); del buf174  # reuse
    cpp_fused_relu_55(c_void_p(buf175.data_ptr()))
    buf176 = buf172; del buf172  # reuse
    # Source Nodes: [layer_outputs_61], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg530_1, reinterpret_tensor(buf175, (128, 512), (512, 1), 0), reinterpret_tensor(arg529_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf176)
    del arg529_1
    del arg530_1
    buf177 = buf173; del buf173  # reuse
    cpp_fused_add_mul_56(c_void_p(buf177.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(arg74_1.data_ptr()), c_void_p(arg75_1.data_ptr()))
    del arg74_1
    del arg75_1
    buf178 = reinterpret_tensor(buf175, (128, 512), (512, 1), 0); del buf175  # reuse
    # Source Nodes: [hidden_states_40], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg532_1, reinterpret_tensor(buf177, (128, 128), (128, 1), 0), reinterpret_tensor(arg531_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf178)
    del arg531_1
    del arg532_1
    buf179 = reinterpret_tensor(buf178, (1, 128, 512), (65536, 512, 1), 0); del buf178  # reuse
    cpp_fused_relu_57(c_void_p(buf179.data_ptr()))
    buf180 = buf176; del buf176  # reuse
    # Source Nodes: [layer_outputs_64], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg534_1, reinterpret_tensor(buf179, (128, 512), (512, 1), 0), reinterpret_tensor(arg533_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf180)
    del arg533_1
    del arg534_1
    buf181 = buf177; del buf177  # reuse
    cpp_fused_add_mul_58(c_void_p(buf181.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(arg76_1.data_ptr()), c_void_p(arg77_1.data_ptr()))
    del arg76_1
    del arg77_1
    buf182 = reinterpret_tensor(buf179, (128, 512), (512, 1), 0); del buf179  # reuse
    # Source Nodes: [hidden_states_42], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg536_1, reinterpret_tensor(buf181, (128, 128), (128, 1), 0), reinterpret_tensor(arg535_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf182)
    del arg535_1
    del arg536_1
    buf183 = reinterpret_tensor(buf182, (1, 128, 512), (65536, 512, 1), 0); del buf182  # reuse
    cpp_fused_relu_59(c_void_p(buf183.data_ptr()))
    buf184 = buf180; del buf180  # reuse
    # Source Nodes: [layer_output_16], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg538_1, reinterpret_tensor(buf183, (128, 512), (512, 1), 0), reinterpret_tensor(arg537_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf184)
    del arg537_1
    del arg538_1
    buf185 = buf181; del buf181  # reuse
    cpp_fused_add_mul_60(c_void_p(buf185.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(arg78_1.data_ptr()), c_void_p(arg79_1.data_ptr()))
    del arg78_1
    del arg79_1
    buf186 = reinterpret_tensor(buf183, (128, 512), (512, 1), 0); del buf183  # reuse
    # Source Nodes: [layer_outputs_67], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg540_1, reinterpret_tensor(buf185, (128, 128), (128, 1), 0), reinterpret_tensor(arg539_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf186)
    del arg539_1
    del arg540_1
    buf187 = buf150; del buf150  # reuse
    cpp_fused_add_mul_61(c_void_p(buf187.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(arg80_1.data_ptr()), c_void_p(arg81_1.data_ptr()))
    del arg80_1
    del arg81_1
    buf188 = reinterpret_tensor(buf185, (128, 128), (128, 1), 0); del buf185  # reuse
    # Source Nodes: [layer_input_27], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg544_1, reinterpret_tensor(buf187, (128, 512), (512, 1), 0), reinterpret_tensor(arg543_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf188)
    del arg543_1
    del arg544_1
    buf189 = reinterpret_tensor(buf188, (1, 128, 128), (16384, 128, 1), 0); del buf188  # reuse
    cpp_fused_add_mul_62(c_void_p(buf189.data_ptr()), c_void_p(arg84_1.data_ptr()), c_void_p(arg85_1.data_ptr()))
    del arg84_1
    del arg85_1
    buf190 = buf184; del buf184  # reuse
    # Source Nodes: [mixed_query_layer_5], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg546_1, reinterpret_tensor(buf189, (128, 128), (128, 1), 0), reinterpret_tensor(arg545_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf190)
    del arg545_1
    del arg546_1
    buf191 = reinterpret_tensor(buf157, (128, 128), (128, 1), 0); del buf157  # reuse
    # Source Nodes: [mixed_key_layer_5], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg548_1, reinterpret_tensor(buf189, (128, 128), (128, 1), 0), reinterpret_tensor(arg547_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf191)
    del arg547_1
    del arg548_1
    buf192 = reinterpret_tensor(buf189, (128, 128), (128, 1), 0); del buf189  # reuse
    # Source Nodes: [mixed_value_layer_5], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg550_1, reinterpret_tensor(buf187, (128, 512), (512, 1), 0), reinterpret_tensor(arg549_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf192)
    del arg549_1
    del arg550_1
    buf193 = reinterpret_tensor(buf190, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf190  # reuse
    buf194 = reinterpret_tensor(buf191, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf191  # reuse
    buf195 = reinterpret_tensor(buf192, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf192  # reuse
    cpp_fused_63(c_void_p(buf193.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf195.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf196 = aten._scaled_dot_product_flash_attention(buf193, buf194, buf195, scale=0.17677669529663687)
    del buf193
    buf197 = buf196[0]
    del buf196
    buf204 = reinterpret_tensor(buf195, (128, 128), (128, 1), 0); del buf195  # reuse
    # Source Nodes: [layer_outputs_70], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg552_1, reinterpret_tensor(buf197, (128, 128), (128, 1), 0), reinterpret_tensor(arg551_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf204)
    del arg551_1
    del arg552_1
    buf205 = reinterpret_tensor(buf197, (128, 128), (128, 1), 0); del buf197  # reuse
    # Source Nodes: [layer_input_25], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg542_1, reinterpret_tensor(buf187, (128, 512), (512, 1), 0), reinterpret_tensor(arg541_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf205)
    del arg541_1
    del arg542_1
    buf206 = reinterpret_tensor(buf204, (1, 128, 128), (16384, 128, 1), 0); del buf204  # reuse
    cpp_fused_add_mul_64(c_void_p(buf206.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(arg82_1.data_ptr()), c_void_p(arg83_1.data_ptr()), c_void_p(arg86_1.data_ptr()), c_void_p(arg87_1.data_ptr()))
    del arg82_1
    del arg83_1
    del arg86_1
    del arg87_1
    buf207 = buf186; del buf186  # reuse
    # Source Nodes: [hidden_states_45], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg554_1, reinterpret_tensor(buf206, (128, 128), (128, 1), 0), reinterpret_tensor(arg553_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf207)
    del arg553_1
    del arg554_1
    buf208 = reinterpret_tensor(buf207, (1, 128, 512), (65536, 512, 1), 0); del buf207  # reuse
    cpp_fused_relu_65(c_void_p(buf208.data_ptr()))
    buf209 = buf205; del buf205  # reuse
    # Source Nodes: [layer_outputs_72], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg556_1, reinterpret_tensor(buf208, (128, 512), (512, 1), 0), reinterpret_tensor(arg555_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf209)
    del arg555_1
    del arg556_1
    buf210 = buf206; del buf206  # reuse
    cpp_fused_add_mul_66(c_void_p(buf210.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(arg88_1.data_ptr()), c_void_p(arg89_1.data_ptr()))
    del arg88_1
    del arg89_1
    buf211 = reinterpret_tensor(buf208, (128, 512), (512, 1), 0); del buf208  # reuse
    # Source Nodes: [hidden_states_47], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg558_1, reinterpret_tensor(buf210, (128, 128), (128, 1), 0), reinterpret_tensor(arg557_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf211)
    del arg557_1
    del arg558_1
    buf212 = reinterpret_tensor(buf211, (1, 128, 512), (65536, 512, 1), 0); del buf211  # reuse
    cpp_fused_relu_67(c_void_p(buf212.data_ptr()))
    buf213 = buf209; del buf209  # reuse
    # Source Nodes: [layer_outputs_75], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg560_1, reinterpret_tensor(buf212, (128, 512), (512, 1), 0), reinterpret_tensor(arg559_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf213)
    del arg559_1
    del arg560_1
    buf214 = buf210; del buf210  # reuse
    cpp_fused_add_mul_68(c_void_p(buf214.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(arg90_1.data_ptr()), c_void_p(arg91_1.data_ptr()))
    del arg90_1
    del arg91_1
    buf215 = reinterpret_tensor(buf212, (128, 512), (512, 1), 0); del buf212  # reuse
    # Source Nodes: [hidden_states_49], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg562_1, reinterpret_tensor(buf214, (128, 128), (128, 1), 0), reinterpret_tensor(arg561_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf215)
    del arg561_1
    del arg562_1
    buf216 = reinterpret_tensor(buf215, (1, 128, 512), (65536, 512, 1), 0); del buf215  # reuse
    cpp_fused_relu_69(c_void_p(buf216.data_ptr()))
    buf217 = buf213; del buf213  # reuse
    # Source Nodes: [layer_outputs_78], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg564_1, reinterpret_tensor(buf216, (128, 512), (512, 1), 0), reinterpret_tensor(arg563_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf217)
    del arg563_1
    del arg564_1
    buf218 = buf214; del buf214  # reuse
    cpp_fused_add_mul_70(c_void_p(buf218.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(arg92_1.data_ptr()), c_void_p(arg93_1.data_ptr()))
    del arg92_1
    del arg93_1
    buf219 = reinterpret_tensor(buf216, (128, 512), (512, 1), 0); del buf216  # reuse
    # Source Nodes: [hidden_states_51], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg566_1, reinterpret_tensor(buf218, (128, 128), (128, 1), 0), reinterpret_tensor(arg565_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf219)
    del arg565_1
    del arg566_1
    buf220 = reinterpret_tensor(buf219, (1, 128, 512), (65536, 512, 1), 0); del buf219  # reuse
    cpp_fused_relu_71(c_void_p(buf220.data_ptr()))
    buf221 = buf217; del buf217  # reuse
    # Source Nodes: [layer_output_20], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg568_1, reinterpret_tensor(buf220, (128, 512), (512, 1), 0), reinterpret_tensor(arg567_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf221)
    del arg567_1
    del arg568_1
    buf222 = buf218; del buf218  # reuse
    cpp_fused_add_mul_72(c_void_p(buf222.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(arg94_1.data_ptr()), c_void_p(arg95_1.data_ptr()))
    del arg94_1
    del arg95_1
    buf223 = reinterpret_tensor(buf220, (128, 512), (512, 1), 0); del buf220  # reuse
    # Source Nodes: [layer_outputs_81], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg570_1, reinterpret_tensor(buf222, (128, 128), (128, 1), 0), reinterpret_tensor(arg569_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf223)
    del arg569_1
    del arg570_1
    buf224 = buf187; del buf187  # reuse
    cpp_fused_add_mul_73(c_void_p(buf224.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(arg96_1.data_ptr()), c_void_p(arg97_1.data_ptr()))
    del arg96_1
    del arg97_1
    buf225 = reinterpret_tensor(buf222, (128, 128), (128, 1), 0); del buf222  # reuse
    # Source Nodes: [layer_input_32], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg574_1, reinterpret_tensor(buf224, (128, 512), (512, 1), 0), reinterpret_tensor(arg573_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf225)
    del arg573_1
    del arg574_1
    buf226 = reinterpret_tensor(buf225, (1, 128, 128), (16384, 128, 1), 0); del buf225  # reuse
    cpp_fused_add_mul_74(c_void_p(buf226.data_ptr()), c_void_p(arg100_1.data_ptr()), c_void_p(arg101_1.data_ptr()))
    del arg100_1
    del arg101_1
    buf227 = buf221; del buf221  # reuse
    # Source Nodes: [mixed_query_layer_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg576_1, reinterpret_tensor(buf226, (128, 128), (128, 1), 0), reinterpret_tensor(arg575_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf227)
    del arg575_1
    del arg576_1
    buf228 = reinterpret_tensor(buf194, (128, 128), (128, 1), 0); del buf194  # reuse
    # Source Nodes: [mixed_key_layer_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg578_1, reinterpret_tensor(buf226, (128, 128), (128, 1), 0), reinterpret_tensor(arg577_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf228)
    del arg577_1
    del arg578_1
    buf229 = reinterpret_tensor(buf226, (128, 128), (128, 1), 0); del buf226  # reuse
    # Source Nodes: [mixed_value_layer_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg580_1, reinterpret_tensor(buf224, (128, 512), (512, 1), 0), reinterpret_tensor(arg579_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf229)
    del arg579_1
    del arg580_1
    buf230 = reinterpret_tensor(buf227, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf227  # reuse
    buf231 = reinterpret_tensor(buf228, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf228  # reuse
    buf232 = reinterpret_tensor(buf229, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf229  # reuse
    cpp_fused_75(c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf232.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf233 = aten._scaled_dot_product_flash_attention(buf230, buf231, buf232, scale=0.17677669529663687)
    del buf230
    buf234 = buf233[0]
    del buf233
    buf241 = reinterpret_tensor(buf232, (128, 128), (128, 1), 0); del buf232  # reuse
    # Source Nodes: [layer_outputs_84], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg582_1, reinterpret_tensor(buf234, (128, 128), (128, 1), 0), reinterpret_tensor(arg581_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf241)
    del arg581_1
    del arg582_1
    buf242 = reinterpret_tensor(buf234, (128, 128), (128, 1), 0); del buf234  # reuse
    # Source Nodes: [layer_input_30], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg572_1, reinterpret_tensor(buf224, (128, 512), (512, 1), 0), reinterpret_tensor(arg571_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf242)
    del arg571_1
    del arg572_1
    buf243 = reinterpret_tensor(buf241, (1, 128, 128), (16384, 128, 1), 0); del buf241  # reuse
    cpp_fused_add_mul_76(c_void_p(buf243.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(arg98_1.data_ptr()), c_void_p(arg99_1.data_ptr()), c_void_p(arg102_1.data_ptr()), c_void_p(arg103_1.data_ptr()))
    del arg102_1
    del arg103_1
    del arg98_1
    del arg99_1
    buf244 = buf223; del buf223  # reuse
    # Source Nodes: [hidden_states_54], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg584_1, reinterpret_tensor(buf243, (128, 128), (128, 1), 0), reinterpret_tensor(arg583_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf244)
    del arg583_1
    del arg584_1
    buf245 = reinterpret_tensor(buf244, (1, 128, 512), (65536, 512, 1), 0); del buf244  # reuse
    cpp_fused_relu_77(c_void_p(buf245.data_ptr()))
    buf246 = buf242; del buf242  # reuse
    # Source Nodes: [layer_outputs_86], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg586_1, reinterpret_tensor(buf245, (128, 512), (512, 1), 0), reinterpret_tensor(arg585_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf246)
    del arg585_1
    del arg586_1
    buf247 = buf243; del buf243  # reuse
    cpp_fused_add_mul_78(c_void_p(buf247.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(arg104_1.data_ptr()), c_void_p(arg105_1.data_ptr()))
    del arg104_1
    del arg105_1
    buf248 = reinterpret_tensor(buf245, (128, 512), (512, 1), 0); del buf245  # reuse
    # Source Nodes: [hidden_states_56], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg588_1, reinterpret_tensor(buf247, (128, 128), (128, 1), 0), reinterpret_tensor(arg587_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf248)
    del arg587_1
    del arg588_1
    buf249 = reinterpret_tensor(buf248, (1, 128, 512), (65536, 512, 1), 0); del buf248  # reuse
    cpp_fused_relu_79(c_void_p(buf249.data_ptr()))
    buf250 = buf246; del buf246  # reuse
    # Source Nodes: [layer_outputs_89], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg590_1, reinterpret_tensor(buf249, (128, 512), (512, 1), 0), reinterpret_tensor(arg589_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf250)
    del arg589_1
    del arg590_1
    buf251 = buf247; del buf247  # reuse
    cpp_fused_add_mul_80(c_void_p(buf251.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(arg106_1.data_ptr()), c_void_p(arg107_1.data_ptr()))
    del arg106_1
    del arg107_1
    buf252 = reinterpret_tensor(buf249, (128, 512), (512, 1), 0); del buf249  # reuse
    # Source Nodes: [hidden_states_58], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg592_1, reinterpret_tensor(buf251, (128, 128), (128, 1), 0), reinterpret_tensor(arg591_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf252)
    del arg591_1
    del arg592_1
    buf253 = reinterpret_tensor(buf252, (1, 128, 512), (65536, 512, 1), 0); del buf252  # reuse
    cpp_fused_relu_81(c_void_p(buf253.data_ptr()))
    buf254 = buf250; del buf250  # reuse
    # Source Nodes: [layer_outputs_92], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg594_1, reinterpret_tensor(buf253, (128, 512), (512, 1), 0), reinterpret_tensor(arg593_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf254)
    del arg593_1
    del arg594_1
    buf255 = buf251; del buf251  # reuse
    cpp_fused_add_mul_82(c_void_p(buf255.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(arg108_1.data_ptr()), c_void_p(arg109_1.data_ptr()))
    del arg108_1
    del arg109_1
    buf256 = reinterpret_tensor(buf253, (128, 512), (512, 1), 0); del buf253  # reuse
    # Source Nodes: [hidden_states_60], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg596_1, reinterpret_tensor(buf255, (128, 128), (128, 1), 0), reinterpret_tensor(arg595_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf256)
    del arg595_1
    del arg596_1
    buf257 = reinterpret_tensor(buf256, (1, 128, 512), (65536, 512, 1), 0); del buf256  # reuse
    cpp_fused_relu_83(c_void_p(buf257.data_ptr()))
    buf258 = buf254; del buf254  # reuse
    # Source Nodes: [layer_output_24], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg598_1, reinterpret_tensor(buf257, (128, 512), (512, 1), 0), reinterpret_tensor(arg597_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf258)
    del arg597_1
    del arg598_1
    buf259 = buf255; del buf255  # reuse
    cpp_fused_add_mul_84(c_void_p(buf259.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(arg110_1.data_ptr()), c_void_p(arg111_1.data_ptr()))
    del arg110_1
    del arg111_1
    buf260 = reinterpret_tensor(buf257, (128, 512), (512, 1), 0); del buf257  # reuse
    # Source Nodes: [layer_outputs_95], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg600_1, reinterpret_tensor(buf259, (128, 128), (128, 1), 0), reinterpret_tensor(arg599_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf260)
    del arg599_1
    del arg600_1
    buf261 = buf224; del buf224  # reuse
    cpp_fused_add_mul_85(c_void_p(buf261.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(arg112_1.data_ptr()), c_void_p(arg113_1.data_ptr()))
    del arg112_1
    del arg113_1
    buf262 = reinterpret_tensor(buf259, (128, 128), (128, 1), 0); del buf259  # reuse
    # Source Nodes: [layer_input_37], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg604_1, reinterpret_tensor(buf261, (128, 512), (512, 1), 0), reinterpret_tensor(arg603_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf262)
    del arg603_1
    del arg604_1
    buf263 = reinterpret_tensor(buf262, (1, 128, 128), (16384, 128, 1), 0); del buf262  # reuse
    cpp_fused_add_mul_86(c_void_p(buf263.data_ptr()), c_void_p(arg116_1.data_ptr()), c_void_p(arg117_1.data_ptr()))
    del arg116_1
    del arg117_1
    buf264 = buf258; del buf258  # reuse
    # Source Nodes: [mixed_query_layer_7], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg606_1, reinterpret_tensor(buf263, (128, 128), (128, 1), 0), reinterpret_tensor(arg605_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf264)
    del arg605_1
    del arg606_1
    buf265 = reinterpret_tensor(buf231, (128, 128), (128, 1), 0); del buf231  # reuse
    # Source Nodes: [mixed_key_layer_7], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg608_1, reinterpret_tensor(buf263, (128, 128), (128, 1), 0), reinterpret_tensor(arg607_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf265)
    del arg607_1
    del arg608_1
    buf266 = reinterpret_tensor(buf263, (128, 128), (128, 1), 0); del buf263  # reuse
    # Source Nodes: [mixed_value_layer_7], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg610_1, reinterpret_tensor(buf261, (128, 512), (512, 1), 0), reinterpret_tensor(arg609_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf266)
    del arg609_1
    del arg610_1
    buf267 = reinterpret_tensor(buf264, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf264  # reuse
    buf268 = reinterpret_tensor(buf265, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf265  # reuse
    buf269 = reinterpret_tensor(buf266, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf266  # reuse
    cpp_fused_87(c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf269.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf270 = aten._scaled_dot_product_flash_attention(buf267, buf268, buf269, scale=0.17677669529663687)
    del buf267
    buf271 = buf270[0]
    del buf270
    buf278 = reinterpret_tensor(buf269, (128, 128), (128, 1), 0); del buf269  # reuse
    # Source Nodes: [layer_outputs_98], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg612_1, reinterpret_tensor(buf271, (128, 128), (128, 1), 0), reinterpret_tensor(arg611_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf278)
    del arg611_1
    del arg612_1
    buf279 = reinterpret_tensor(buf271, (128, 128), (128, 1), 0); del buf271  # reuse
    # Source Nodes: [layer_input_35], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg602_1, reinterpret_tensor(buf261, (128, 512), (512, 1), 0), reinterpret_tensor(arg601_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf279)
    del arg601_1
    del arg602_1
    buf280 = reinterpret_tensor(buf278, (1, 128, 128), (16384, 128, 1), 0); del buf278  # reuse
    cpp_fused_add_mul_88(c_void_p(buf280.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(arg114_1.data_ptr()), c_void_p(arg115_1.data_ptr()), c_void_p(arg118_1.data_ptr()), c_void_p(arg119_1.data_ptr()))
    del arg114_1
    del arg115_1
    del arg118_1
    del arg119_1
    buf281 = buf260; del buf260  # reuse
    # Source Nodes: [hidden_states_63], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg614_1, reinterpret_tensor(buf280, (128, 128), (128, 1), 0), reinterpret_tensor(arg613_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf281)
    del arg613_1
    del arg614_1
    buf282 = reinterpret_tensor(buf281, (1, 128, 512), (65536, 512, 1), 0); del buf281  # reuse
    cpp_fused_relu_89(c_void_p(buf282.data_ptr()))
    buf283 = buf279; del buf279  # reuse
    # Source Nodes: [layer_outputs_100], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg616_1, reinterpret_tensor(buf282, (128, 512), (512, 1), 0), reinterpret_tensor(arg615_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf283)
    del arg615_1
    del arg616_1
    buf284 = buf280; del buf280  # reuse
    cpp_fused_add_mul_90(c_void_p(buf284.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(arg120_1.data_ptr()), c_void_p(arg121_1.data_ptr()))
    del arg120_1
    del arg121_1
    buf285 = reinterpret_tensor(buf282, (128, 512), (512, 1), 0); del buf282  # reuse
    # Source Nodes: [hidden_states_65], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg618_1, reinterpret_tensor(buf284, (128, 128), (128, 1), 0), reinterpret_tensor(arg617_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf285)
    del arg617_1
    del arg618_1
    buf286 = reinterpret_tensor(buf285, (1, 128, 512), (65536, 512, 1), 0); del buf285  # reuse
    cpp_fused_relu_91(c_void_p(buf286.data_ptr()))
    buf287 = buf283; del buf283  # reuse
    # Source Nodes: [layer_outputs_103], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg620_1, reinterpret_tensor(buf286, (128, 512), (512, 1), 0), reinterpret_tensor(arg619_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf287)
    del arg619_1
    del arg620_1
    buf288 = buf284; del buf284  # reuse
    cpp_fused_add_mul_92(c_void_p(buf288.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(arg122_1.data_ptr()), c_void_p(arg123_1.data_ptr()))
    del arg122_1
    del arg123_1
    buf289 = reinterpret_tensor(buf286, (128, 512), (512, 1), 0); del buf286  # reuse
    # Source Nodes: [hidden_states_67], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg622_1, reinterpret_tensor(buf288, (128, 128), (128, 1), 0), reinterpret_tensor(arg621_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf289)
    del arg621_1
    del arg622_1
    buf290 = reinterpret_tensor(buf289, (1, 128, 512), (65536, 512, 1), 0); del buf289  # reuse
    cpp_fused_relu_93(c_void_p(buf290.data_ptr()))
    buf291 = buf287; del buf287  # reuse
    # Source Nodes: [layer_outputs_106], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg624_1, reinterpret_tensor(buf290, (128, 512), (512, 1), 0), reinterpret_tensor(arg623_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf291)
    del arg623_1
    del arg624_1
    buf292 = buf288; del buf288  # reuse
    cpp_fused_add_mul_94(c_void_p(buf292.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(arg124_1.data_ptr()), c_void_p(arg125_1.data_ptr()))
    del arg124_1
    del arg125_1
    buf293 = reinterpret_tensor(buf290, (128, 512), (512, 1), 0); del buf290  # reuse
    # Source Nodes: [hidden_states_69], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg626_1, reinterpret_tensor(buf292, (128, 128), (128, 1), 0), reinterpret_tensor(arg625_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf293)
    del arg625_1
    del arg626_1
    buf294 = reinterpret_tensor(buf293, (1, 128, 512), (65536, 512, 1), 0); del buf293  # reuse
    cpp_fused_relu_95(c_void_p(buf294.data_ptr()))
    buf295 = buf291; del buf291  # reuse
    # Source Nodes: [layer_output_28], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg628_1, reinterpret_tensor(buf294, (128, 512), (512, 1), 0), reinterpret_tensor(arg627_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf295)
    del arg627_1
    del arg628_1
    buf296 = buf292; del buf292  # reuse
    cpp_fused_add_mul_96(c_void_p(buf296.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(arg126_1.data_ptr()), c_void_p(arg127_1.data_ptr()))
    del arg126_1
    del arg127_1
    buf297 = reinterpret_tensor(buf294, (128, 512), (512, 1), 0); del buf294  # reuse
    # Source Nodes: [layer_outputs_109], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg630_1, reinterpret_tensor(buf296, (128, 128), (128, 1), 0), reinterpret_tensor(arg629_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf297)
    del arg629_1
    del arg630_1
    buf298 = buf261; del buf261  # reuse
    cpp_fused_add_mul_97(c_void_p(buf298.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(arg128_1.data_ptr()), c_void_p(arg129_1.data_ptr()))
    del arg128_1
    del arg129_1
    buf299 = reinterpret_tensor(buf296, (128, 128), (128, 1), 0); del buf296  # reuse
    # Source Nodes: [layer_input_42], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg634_1, reinterpret_tensor(buf298, (128, 512), (512, 1), 0), reinterpret_tensor(arg633_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf299)
    del arg633_1
    del arg634_1
    buf300 = reinterpret_tensor(buf299, (1, 128, 128), (16384, 128, 1), 0); del buf299  # reuse
    cpp_fused_add_mul_98(c_void_p(buf300.data_ptr()), c_void_p(arg132_1.data_ptr()), c_void_p(arg133_1.data_ptr()))
    del arg132_1
    del arg133_1
    buf301 = buf295; del buf295  # reuse
    # Source Nodes: [mixed_query_layer_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg636_1, reinterpret_tensor(buf300, (128, 128), (128, 1), 0), reinterpret_tensor(arg635_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf301)
    del arg635_1
    del arg636_1
    buf302 = reinterpret_tensor(buf268, (128, 128), (128, 1), 0); del buf268  # reuse
    # Source Nodes: [mixed_key_layer_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg638_1, reinterpret_tensor(buf300, (128, 128), (128, 1), 0), reinterpret_tensor(arg637_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf302)
    del arg637_1
    del arg638_1
    buf303 = reinterpret_tensor(buf300, (128, 128), (128, 1), 0); del buf300  # reuse
    # Source Nodes: [mixed_value_layer_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg640_1, reinterpret_tensor(buf298, (128, 512), (512, 1), 0), reinterpret_tensor(arg639_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf303)
    del arg639_1
    del arg640_1
    buf304 = reinterpret_tensor(buf301, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf301  # reuse
    buf305 = reinterpret_tensor(buf302, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf302  # reuse
    buf306 = reinterpret_tensor(buf303, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf303  # reuse
    cpp_fused_99(c_void_p(buf304.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf306.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf307 = aten._scaled_dot_product_flash_attention(buf304, buf305, buf306, scale=0.17677669529663687)
    del buf304
    buf308 = buf307[0]
    del buf307
    buf315 = reinterpret_tensor(buf306, (128, 128), (128, 1), 0); del buf306  # reuse
    # Source Nodes: [layer_outputs_112], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg642_1, reinterpret_tensor(buf308, (128, 128), (128, 1), 0), reinterpret_tensor(arg641_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf315)
    del arg641_1
    del arg642_1
    buf316 = reinterpret_tensor(buf308, (128, 128), (128, 1), 0); del buf308  # reuse
    # Source Nodes: [layer_input_40], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg632_1, reinterpret_tensor(buf298, (128, 512), (512, 1), 0), reinterpret_tensor(arg631_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf316)
    del arg631_1
    del arg632_1
    buf317 = reinterpret_tensor(buf315, (1, 128, 128), (16384, 128, 1), 0); del buf315  # reuse
    cpp_fused_add_mul_100(c_void_p(buf317.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(arg130_1.data_ptr()), c_void_p(arg131_1.data_ptr()), c_void_p(arg134_1.data_ptr()), c_void_p(arg135_1.data_ptr()))
    del arg130_1
    del arg131_1
    del arg134_1
    del arg135_1
    buf318 = buf297; del buf297  # reuse
    # Source Nodes: [hidden_states_72], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg644_1, reinterpret_tensor(buf317, (128, 128), (128, 1), 0), reinterpret_tensor(arg643_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf318)
    del arg643_1
    del arg644_1
    buf319 = reinterpret_tensor(buf318, (1, 128, 512), (65536, 512, 1), 0); del buf318  # reuse
    cpp_fused_relu_101(c_void_p(buf319.data_ptr()))
    buf320 = buf316; del buf316  # reuse
    # Source Nodes: [layer_outputs_114], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg646_1, reinterpret_tensor(buf319, (128, 512), (512, 1), 0), reinterpret_tensor(arg645_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf320)
    del arg645_1
    del arg646_1
    buf321 = buf317; del buf317  # reuse
    cpp_fused_add_mul_102(c_void_p(buf321.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(arg136_1.data_ptr()), c_void_p(arg137_1.data_ptr()))
    del arg136_1
    del arg137_1
    buf322 = reinterpret_tensor(buf319, (128, 512), (512, 1), 0); del buf319  # reuse
    # Source Nodes: [hidden_states_74], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg648_1, reinterpret_tensor(buf321, (128, 128), (128, 1), 0), reinterpret_tensor(arg647_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf322)
    del arg647_1
    del arg648_1
    buf323 = reinterpret_tensor(buf322, (1, 128, 512), (65536, 512, 1), 0); del buf322  # reuse
    cpp_fused_relu_103(c_void_p(buf323.data_ptr()))
    buf324 = buf320; del buf320  # reuse
    # Source Nodes: [layer_outputs_117], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg650_1, reinterpret_tensor(buf323, (128, 512), (512, 1), 0), reinterpret_tensor(arg649_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf324)
    del arg649_1
    del arg650_1
    buf325 = buf321; del buf321  # reuse
    cpp_fused_add_mul_104(c_void_p(buf325.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(arg138_1.data_ptr()), c_void_p(arg139_1.data_ptr()))
    del arg138_1
    del arg139_1
    buf326 = reinterpret_tensor(buf323, (128, 512), (512, 1), 0); del buf323  # reuse
    # Source Nodes: [hidden_states_76], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg652_1, reinterpret_tensor(buf325, (128, 128), (128, 1), 0), reinterpret_tensor(arg651_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf326)
    del arg651_1
    del arg652_1
    buf327 = reinterpret_tensor(buf326, (1, 128, 512), (65536, 512, 1), 0); del buf326  # reuse
    cpp_fused_relu_105(c_void_p(buf327.data_ptr()))
    buf328 = buf324; del buf324  # reuse
    # Source Nodes: [layer_outputs_120], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg654_1, reinterpret_tensor(buf327, (128, 512), (512, 1), 0), reinterpret_tensor(arg653_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf328)
    del arg653_1
    del arg654_1
    buf329 = buf325; del buf325  # reuse
    cpp_fused_add_mul_106(c_void_p(buf329.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(arg140_1.data_ptr()), c_void_p(arg141_1.data_ptr()))
    del arg140_1
    del arg141_1
    buf330 = reinterpret_tensor(buf327, (128, 512), (512, 1), 0); del buf327  # reuse
    # Source Nodes: [hidden_states_78], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg656_1, reinterpret_tensor(buf329, (128, 128), (128, 1), 0), reinterpret_tensor(arg655_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf330)
    del arg655_1
    del arg656_1
    buf331 = reinterpret_tensor(buf330, (1, 128, 512), (65536, 512, 1), 0); del buf330  # reuse
    cpp_fused_relu_107(c_void_p(buf331.data_ptr()))
    buf332 = buf328; del buf328  # reuse
    # Source Nodes: [layer_output_32], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg658_1, reinterpret_tensor(buf331, (128, 512), (512, 1), 0), reinterpret_tensor(arg657_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf332)
    del arg657_1
    del arg658_1
    buf333 = buf329; del buf329  # reuse
    cpp_fused_add_mul_108(c_void_p(buf333.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(arg142_1.data_ptr()), c_void_p(arg143_1.data_ptr()))
    del arg142_1
    del arg143_1
    buf334 = reinterpret_tensor(buf331, (128, 512), (512, 1), 0); del buf331  # reuse
    # Source Nodes: [layer_outputs_123], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg660_1, reinterpret_tensor(buf333, (128, 128), (128, 1), 0), reinterpret_tensor(arg659_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf334)
    del arg659_1
    del arg660_1
    buf335 = buf298; del buf298  # reuse
    cpp_fused_add_mul_109(c_void_p(buf335.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(arg144_1.data_ptr()), c_void_p(arg145_1.data_ptr()))
    del arg144_1
    del arg145_1
    buf336 = reinterpret_tensor(buf333, (128, 128), (128, 1), 0); del buf333  # reuse
    # Source Nodes: [layer_input_47], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg664_1, reinterpret_tensor(buf335, (128, 512), (512, 1), 0), reinterpret_tensor(arg663_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf336)
    del arg663_1
    del arg664_1
    buf337 = reinterpret_tensor(buf336, (1, 128, 128), (16384, 128, 1), 0); del buf336  # reuse
    cpp_fused_add_mul_110(c_void_p(buf337.data_ptr()), c_void_p(arg148_1.data_ptr()), c_void_p(arg149_1.data_ptr()))
    del arg148_1
    del arg149_1
    buf338 = buf332; del buf332  # reuse
    # Source Nodes: [mixed_query_layer_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg666_1, reinterpret_tensor(buf337, (128, 128), (128, 1), 0), reinterpret_tensor(arg665_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf338)
    del arg665_1
    del arg666_1
    buf339 = reinterpret_tensor(buf305, (128, 128), (128, 1), 0); del buf305  # reuse
    # Source Nodes: [mixed_key_layer_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg668_1, reinterpret_tensor(buf337, (128, 128), (128, 1), 0), reinterpret_tensor(arg667_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf339)
    del arg667_1
    del arg668_1
    buf340 = reinterpret_tensor(buf337, (128, 128), (128, 1), 0); del buf337  # reuse
    # Source Nodes: [mixed_value_layer_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg670_1, reinterpret_tensor(buf335, (128, 512), (512, 1), 0), reinterpret_tensor(arg669_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf340)
    del arg669_1
    del arg670_1
    buf341 = reinterpret_tensor(buf338, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf338  # reuse
    buf342 = reinterpret_tensor(buf339, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf339  # reuse
    buf343 = reinterpret_tensor(buf340, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf340  # reuse
    cpp_fused_111(c_void_p(buf341.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf343.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf344 = aten._scaled_dot_product_flash_attention(buf341, buf342, buf343, scale=0.17677669529663687)
    del buf341
    buf345 = buf344[0]
    del buf344
    buf352 = reinterpret_tensor(buf343, (128, 128), (128, 1), 0); del buf343  # reuse
    # Source Nodes: [layer_outputs_126], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg672_1, reinterpret_tensor(buf345, (128, 128), (128, 1), 0), reinterpret_tensor(arg671_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf352)
    del arg671_1
    del arg672_1
    buf353 = reinterpret_tensor(buf345, (128, 128), (128, 1), 0); del buf345  # reuse
    # Source Nodes: [layer_input_45], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg662_1, reinterpret_tensor(buf335, (128, 512), (512, 1), 0), reinterpret_tensor(arg661_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf353)
    del arg661_1
    del arg662_1
    buf354 = reinterpret_tensor(buf352, (1, 128, 128), (16384, 128, 1), 0); del buf352  # reuse
    cpp_fused_add_mul_112(c_void_p(buf354.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(arg146_1.data_ptr()), c_void_p(arg147_1.data_ptr()), c_void_p(arg150_1.data_ptr()), c_void_p(arg151_1.data_ptr()))
    del arg146_1
    del arg147_1
    del arg150_1
    del arg151_1
    buf355 = buf334; del buf334  # reuse
    # Source Nodes: [hidden_states_81], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg674_1, reinterpret_tensor(buf354, (128, 128), (128, 1), 0), reinterpret_tensor(arg673_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf355)
    del arg673_1
    del arg674_1
    buf356 = reinterpret_tensor(buf355, (1, 128, 512), (65536, 512, 1), 0); del buf355  # reuse
    cpp_fused_relu_113(c_void_p(buf356.data_ptr()))
    buf357 = buf353; del buf353  # reuse
    # Source Nodes: [layer_outputs_128], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg676_1, reinterpret_tensor(buf356, (128, 512), (512, 1), 0), reinterpret_tensor(arg675_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf357)
    del arg675_1
    del arg676_1
    buf358 = buf354; del buf354  # reuse
    cpp_fused_add_mul_114(c_void_p(buf358.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(arg152_1.data_ptr()), c_void_p(arg153_1.data_ptr()))
    del arg152_1
    del arg153_1
    buf359 = reinterpret_tensor(buf356, (128, 512), (512, 1), 0); del buf356  # reuse
    # Source Nodes: [hidden_states_83], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg678_1, reinterpret_tensor(buf358, (128, 128), (128, 1), 0), reinterpret_tensor(arg677_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf359)
    del arg677_1
    del arg678_1
    buf360 = reinterpret_tensor(buf359, (1, 128, 512), (65536, 512, 1), 0); del buf359  # reuse
    cpp_fused_relu_115(c_void_p(buf360.data_ptr()))
    buf361 = buf357; del buf357  # reuse
    # Source Nodes: [layer_outputs_131], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg680_1, reinterpret_tensor(buf360, (128, 512), (512, 1), 0), reinterpret_tensor(arg679_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf361)
    del arg679_1
    del arg680_1
    buf362 = buf358; del buf358  # reuse
    cpp_fused_add_mul_116(c_void_p(buf362.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(arg154_1.data_ptr()), c_void_p(arg155_1.data_ptr()))
    del arg154_1
    del arg155_1
    buf363 = reinterpret_tensor(buf360, (128, 512), (512, 1), 0); del buf360  # reuse
    # Source Nodes: [hidden_states_85], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg682_1, reinterpret_tensor(buf362, (128, 128), (128, 1), 0), reinterpret_tensor(arg681_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf363)
    del arg681_1
    del arg682_1
    buf364 = reinterpret_tensor(buf363, (1, 128, 512), (65536, 512, 1), 0); del buf363  # reuse
    cpp_fused_relu_117(c_void_p(buf364.data_ptr()))
    buf365 = buf361; del buf361  # reuse
    # Source Nodes: [layer_outputs_134], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg684_1, reinterpret_tensor(buf364, (128, 512), (512, 1), 0), reinterpret_tensor(arg683_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf365)
    del arg683_1
    del arg684_1
    buf366 = buf362; del buf362  # reuse
    cpp_fused_add_mul_118(c_void_p(buf366.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(arg156_1.data_ptr()), c_void_p(arg157_1.data_ptr()))
    del arg156_1
    del arg157_1
    buf367 = reinterpret_tensor(buf364, (128, 512), (512, 1), 0); del buf364  # reuse
    # Source Nodes: [hidden_states_87], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg686_1, reinterpret_tensor(buf366, (128, 128), (128, 1), 0), reinterpret_tensor(arg685_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf367)
    del arg685_1
    del arg686_1
    buf368 = reinterpret_tensor(buf367, (1, 128, 512), (65536, 512, 1), 0); del buf367  # reuse
    cpp_fused_relu_119(c_void_p(buf368.data_ptr()))
    buf369 = buf365; del buf365  # reuse
    # Source Nodes: [layer_output_36], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg688_1, reinterpret_tensor(buf368, (128, 512), (512, 1), 0), reinterpret_tensor(arg687_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf369)
    del arg687_1
    del arg688_1
    buf370 = buf366; del buf366  # reuse
    cpp_fused_add_mul_120(c_void_p(buf370.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(arg158_1.data_ptr()), c_void_p(arg159_1.data_ptr()))
    del arg158_1
    del arg159_1
    buf371 = reinterpret_tensor(buf368, (128, 512), (512, 1), 0); del buf368  # reuse
    # Source Nodes: [layer_outputs_137], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg690_1, reinterpret_tensor(buf370, (128, 128), (128, 1), 0), reinterpret_tensor(arg689_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf371)
    del arg689_1
    del arg690_1
    buf372 = buf335; del buf335  # reuse
    cpp_fused_add_mul_121(c_void_p(buf372.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(arg160_1.data_ptr()), c_void_p(arg161_1.data_ptr()))
    del arg160_1
    del arg161_1
    buf373 = reinterpret_tensor(buf370, (128, 128), (128, 1), 0); del buf370  # reuse
    # Source Nodes: [layer_input_52], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg694_1, reinterpret_tensor(buf372, (128, 512), (512, 1), 0), reinterpret_tensor(arg693_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf373)
    del arg693_1
    del arg694_1
    buf374 = reinterpret_tensor(buf373, (1, 128, 128), (16384, 128, 1), 0); del buf373  # reuse
    cpp_fused_add_mul_122(c_void_p(buf374.data_ptr()), c_void_p(arg164_1.data_ptr()), c_void_p(arg165_1.data_ptr()))
    del arg164_1
    del arg165_1
    buf375 = buf369; del buf369  # reuse
    # Source Nodes: [mixed_query_layer_10], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg696_1, reinterpret_tensor(buf374, (128, 128), (128, 1), 0), reinterpret_tensor(arg695_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf375)
    del arg695_1
    del arg696_1
    buf376 = reinterpret_tensor(buf342, (128, 128), (128, 1), 0); del buf342  # reuse
    # Source Nodes: [mixed_key_layer_10], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg698_1, reinterpret_tensor(buf374, (128, 128), (128, 1), 0), reinterpret_tensor(arg697_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf376)
    del arg697_1
    del arg698_1
    buf377 = reinterpret_tensor(buf374, (128, 128), (128, 1), 0); del buf374  # reuse
    # Source Nodes: [mixed_value_layer_10], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg700_1, reinterpret_tensor(buf372, (128, 512), (512, 1), 0), reinterpret_tensor(arg699_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf377)
    del arg699_1
    del arg700_1
    buf378 = reinterpret_tensor(buf375, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf375  # reuse
    buf379 = reinterpret_tensor(buf376, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf376  # reuse
    buf380 = reinterpret_tensor(buf377, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf377  # reuse
    cpp_fused_123(c_void_p(buf378.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf380.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf381 = aten._scaled_dot_product_flash_attention(buf378, buf379, buf380, scale=0.17677669529663687)
    del buf378
    buf382 = buf381[0]
    del buf381
    buf389 = reinterpret_tensor(buf380, (128, 128), (128, 1), 0); del buf380  # reuse
    # Source Nodes: [layer_outputs_140], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg702_1, reinterpret_tensor(buf382, (128, 128), (128, 1), 0), reinterpret_tensor(arg701_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf389)
    del arg701_1
    del arg702_1
    buf390 = reinterpret_tensor(buf382, (128, 128), (128, 1), 0); del buf382  # reuse
    # Source Nodes: [layer_input_50], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg692_1, reinterpret_tensor(buf372, (128, 512), (512, 1), 0), reinterpret_tensor(arg691_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf390)
    del arg691_1
    del arg692_1
    buf391 = reinterpret_tensor(buf389, (1, 128, 128), (16384, 128, 1), 0); del buf389  # reuse
    cpp_fused_add_mul_124(c_void_p(buf391.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(arg162_1.data_ptr()), c_void_p(arg163_1.data_ptr()), c_void_p(arg166_1.data_ptr()), c_void_p(arg167_1.data_ptr()))
    del arg162_1
    del arg163_1
    del arg166_1
    del arg167_1
    buf392 = buf371; del buf371  # reuse
    # Source Nodes: [hidden_states_90], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg704_1, reinterpret_tensor(buf391, (128, 128), (128, 1), 0), reinterpret_tensor(arg703_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf392)
    del arg703_1
    del arg704_1
    buf393 = reinterpret_tensor(buf392, (1, 128, 512), (65536, 512, 1), 0); del buf392  # reuse
    cpp_fused_relu_125(c_void_p(buf393.data_ptr()))
    buf394 = buf390; del buf390  # reuse
    # Source Nodes: [layer_outputs_142], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg706_1, reinterpret_tensor(buf393, (128, 512), (512, 1), 0), reinterpret_tensor(arg705_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf394)
    del arg705_1
    del arg706_1
    buf395 = buf391; del buf391  # reuse
    cpp_fused_add_mul_126(c_void_p(buf395.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(arg168_1.data_ptr()), c_void_p(arg169_1.data_ptr()))
    del arg168_1
    del arg169_1
    buf396 = reinterpret_tensor(buf393, (128, 512), (512, 1), 0); del buf393  # reuse
    # Source Nodes: [hidden_states_92], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg708_1, reinterpret_tensor(buf395, (128, 128), (128, 1), 0), reinterpret_tensor(arg707_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf396)
    del arg707_1
    del arg708_1
    buf397 = reinterpret_tensor(buf396, (1, 128, 512), (65536, 512, 1), 0); del buf396  # reuse
    cpp_fused_relu_127(c_void_p(buf397.data_ptr()))
    buf398 = buf394; del buf394  # reuse
    # Source Nodes: [layer_outputs_145], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg710_1, reinterpret_tensor(buf397, (128, 512), (512, 1), 0), reinterpret_tensor(arg709_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf398)
    del arg709_1
    del arg710_1
    buf399 = buf395; del buf395  # reuse
    cpp_fused_add_mul_128(c_void_p(buf399.data_ptr()), c_void_p(buf398.data_ptr()), c_void_p(arg170_1.data_ptr()), c_void_p(arg171_1.data_ptr()))
    del arg170_1
    del arg171_1
    buf400 = reinterpret_tensor(buf397, (128, 512), (512, 1), 0); del buf397  # reuse
    # Source Nodes: [hidden_states_94], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg712_1, reinterpret_tensor(buf399, (128, 128), (128, 1), 0), reinterpret_tensor(arg711_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf400)
    del arg711_1
    del arg712_1
    buf401 = reinterpret_tensor(buf400, (1, 128, 512), (65536, 512, 1), 0); del buf400  # reuse
    cpp_fused_relu_129(c_void_p(buf401.data_ptr()))
    buf402 = buf398; del buf398  # reuse
    # Source Nodes: [layer_outputs_148], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg714_1, reinterpret_tensor(buf401, (128, 512), (512, 1), 0), reinterpret_tensor(arg713_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf402)
    del arg713_1
    del arg714_1
    buf403 = buf399; del buf399  # reuse
    cpp_fused_add_mul_130(c_void_p(buf403.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(arg172_1.data_ptr()), c_void_p(arg173_1.data_ptr()))
    del arg172_1
    del arg173_1
    buf404 = reinterpret_tensor(buf401, (128, 512), (512, 1), 0); del buf401  # reuse
    # Source Nodes: [hidden_states_96], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg716_1, reinterpret_tensor(buf403, (128, 128), (128, 1), 0), reinterpret_tensor(arg715_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf404)
    del arg715_1
    del arg716_1
    buf405 = reinterpret_tensor(buf404, (1, 128, 512), (65536, 512, 1), 0); del buf404  # reuse
    cpp_fused_relu_131(c_void_p(buf405.data_ptr()))
    buf406 = buf402; del buf402  # reuse
    # Source Nodes: [layer_output_40], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg718_1, reinterpret_tensor(buf405, (128, 512), (512, 1), 0), reinterpret_tensor(arg717_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf406)
    del arg717_1
    del arg718_1
    buf407 = buf403; del buf403  # reuse
    cpp_fused_add_mul_132(c_void_p(buf407.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(arg174_1.data_ptr()), c_void_p(arg175_1.data_ptr()))
    del arg174_1
    del arg175_1
    buf408 = reinterpret_tensor(buf405, (128, 512), (512, 1), 0); del buf405  # reuse
    # Source Nodes: [layer_outputs_151], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg720_1, reinterpret_tensor(buf407, (128, 128), (128, 1), 0), reinterpret_tensor(arg719_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf408)
    del arg719_1
    del arg720_1
    buf409 = buf372; del buf372  # reuse
    cpp_fused_add_mul_133(c_void_p(buf409.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(arg176_1.data_ptr()), c_void_p(arg177_1.data_ptr()))
    del arg176_1
    del arg177_1
    buf410 = reinterpret_tensor(buf407, (128, 128), (128, 1), 0); del buf407  # reuse
    # Source Nodes: [layer_input_57], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg724_1, reinterpret_tensor(buf409, (128, 512), (512, 1), 0), reinterpret_tensor(arg723_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf410)
    del arg723_1
    del arg724_1
    buf411 = reinterpret_tensor(buf410, (1, 128, 128), (16384, 128, 1), 0); del buf410  # reuse
    cpp_fused_add_mul_134(c_void_p(buf411.data_ptr()), c_void_p(arg180_1.data_ptr()), c_void_p(arg181_1.data_ptr()))
    del arg180_1
    del arg181_1
    buf412 = buf406; del buf406  # reuse
    # Source Nodes: [mixed_query_layer_11], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg726_1, reinterpret_tensor(buf411, (128, 128), (128, 1), 0), reinterpret_tensor(arg725_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf412)
    del arg725_1
    del arg726_1
    buf413 = reinterpret_tensor(buf379, (128, 128), (128, 1), 0); del buf379  # reuse
    # Source Nodes: [mixed_key_layer_11], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg728_1, reinterpret_tensor(buf411, (128, 128), (128, 1), 0), reinterpret_tensor(arg727_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf413)
    del arg727_1
    del arg728_1
    buf414 = reinterpret_tensor(buf411, (128, 128), (128, 1), 0); del buf411  # reuse
    # Source Nodes: [mixed_value_layer_11], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg730_1, reinterpret_tensor(buf409, (128, 512), (512, 1), 0), reinterpret_tensor(arg729_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf414)
    del arg729_1
    del arg730_1
    buf415 = reinterpret_tensor(buf412, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf412  # reuse
    buf416 = reinterpret_tensor(buf413, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf413  # reuse
    buf417 = reinterpret_tensor(buf414, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf414  # reuse
    cpp_fused_135(c_void_p(buf415.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(buf417.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf418 = aten._scaled_dot_product_flash_attention(buf415, buf416, buf417, scale=0.17677669529663687)
    del buf415
    buf419 = buf418[0]
    del buf418
    buf426 = reinterpret_tensor(buf417, (128, 128), (128, 1), 0); del buf417  # reuse
    # Source Nodes: [layer_outputs_154], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg732_1, reinterpret_tensor(buf419, (128, 128), (128, 1), 0), reinterpret_tensor(arg731_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf426)
    del arg731_1
    del arg732_1
    buf427 = reinterpret_tensor(buf419, (128, 128), (128, 1), 0); del buf419  # reuse
    # Source Nodes: [layer_input_55], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg722_1, reinterpret_tensor(buf409, (128, 512), (512, 1), 0), reinterpret_tensor(arg721_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf427)
    del arg721_1
    del arg722_1
    buf428 = reinterpret_tensor(buf426, (1, 128, 128), (16384, 128, 1), 0); del buf426  # reuse
    cpp_fused_add_mul_136(c_void_p(buf428.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(arg178_1.data_ptr()), c_void_p(arg179_1.data_ptr()), c_void_p(arg182_1.data_ptr()), c_void_p(arg183_1.data_ptr()))
    del arg178_1
    del arg179_1
    del arg182_1
    del arg183_1
    buf429 = buf408; del buf408  # reuse
    # Source Nodes: [hidden_states_99], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg734_1, reinterpret_tensor(buf428, (128, 128), (128, 1), 0), reinterpret_tensor(arg733_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf429)
    del arg733_1
    del arg734_1
    buf430 = reinterpret_tensor(buf429, (1, 128, 512), (65536, 512, 1), 0); del buf429  # reuse
    cpp_fused_relu_137(c_void_p(buf430.data_ptr()))
    buf431 = buf427; del buf427  # reuse
    # Source Nodes: [layer_outputs_156], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg736_1, reinterpret_tensor(buf430, (128, 512), (512, 1), 0), reinterpret_tensor(arg735_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf431)
    del arg735_1
    del arg736_1
    buf432 = buf428; del buf428  # reuse
    cpp_fused_add_mul_138(c_void_p(buf432.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(arg184_1.data_ptr()), c_void_p(arg185_1.data_ptr()))
    del arg184_1
    del arg185_1
    buf433 = reinterpret_tensor(buf430, (128, 512), (512, 1), 0); del buf430  # reuse
    # Source Nodes: [hidden_states_101], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg738_1, reinterpret_tensor(buf432, (128, 128), (128, 1), 0), reinterpret_tensor(arg737_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf433)
    del arg737_1
    del arg738_1
    buf434 = reinterpret_tensor(buf433, (1, 128, 512), (65536, 512, 1), 0); del buf433  # reuse
    cpp_fused_relu_139(c_void_p(buf434.data_ptr()))
    buf435 = buf431; del buf431  # reuse
    # Source Nodes: [layer_outputs_159], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg740_1, reinterpret_tensor(buf434, (128, 512), (512, 1), 0), reinterpret_tensor(arg739_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf435)
    del arg739_1
    del arg740_1
    buf436 = buf432; del buf432  # reuse
    cpp_fused_add_mul_140(c_void_p(buf436.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(arg186_1.data_ptr()), c_void_p(arg187_1.data_ptr()))
    del arg186_1
    del arg187_1
    buf437 = reinterpret_tensor(buf434, (128, 512), (512, 1), 0); del buf434  # reuse
    # Source Nodes: [hidden_states_103], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg742_1, reinterpret_tensor(buf436, (128, 128), (128, 1), 0), reinterpret_tensor(arg741_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf437)
    del arg741_1
    del arg742_1
    buf438 = reinterpret_tensor(buf437, (1, 128, 512), (65536, 512, 1), 0); del buf437  # reuse
    cpp_fused_relu_141(c_void_p(buf438.data_ptr()))
    buf439 = buf435; del buf435  # reuse
    # Source Nodes: [layer_outputs_162], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg744_1, reinterpret_tensor(buf438, (128, 512), (512, 1), 0), reinterpret_tensor(arg743_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf439)
    del arg743_1
    del arg744_1
    buf440 = buf436; del buf436  # reuse
    cpp_fused_add_mul_142(c_void_p(buf440.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(arg188_1.data_ptr()), c_void_p(arg189_1.data_ptr()))
    del arg188_1
    del arg189_1
    buf441 = reinterpret_tensor(buf438, (128, 512), (512, 1), 0); del buf438  # reuse
    # Source Nodes: [hidden_states_105], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg746_1, reinterpret_tensor(buf440, (128, 128), (128, 1), 0), reinterpret_tensor(arg745_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf441)
    del arg745_1
    del arg746_1
    buf442 = reinterpret_tensor(buf441, (1, 128, 512), (65536, 512, 1), 0); del buf441  # reuse
    cpp_fused_relu_143(c_void_p(buf442.data_ptr()))
    buf443 = buf439; del buf439  # reuse
    # Source Nodes: [layer_output_44], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg748_1, reinterpret_tensor(buf442, (128, 512), (512, 1), 0), reinterpret_tensor(arg747_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf443)
    del arg747_1
    del arg748_1
    buf444 = buf440; del buf440  # reuse
    cpp_fused_add_mul_144(c_void_p(buf444.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(arg190_1.data_ptr()), c_void_p(arg191_1.data_ptr()))
    del arg190_1
    del arg191_1
    buf445 = reinterpret_tensor(buf442, (128, 512), (512, 1), 0); del buf442  # reuse
    # Source Nodes: [layer_outputs_165], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg750_1, reinterpret_tensor(buf444, (128, 128), (128, 1), 0), reinterpret_tensor(arg749_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf445)
    del arg749_1
    del arg750_1
    buf446 = buf409; del buf409  # reuse
    cpp_fused_add_mul_145(c_void_p(buf446.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(arg192_1.data_ptr()), c_void_p(arg193_1.data_ptr()))
    del arg192_1
    del arg193_1
    buf447 = reinterpret_tensor(buf444, (128, 128), (128, 1), 0); del buf444  # reuse
    # Source Nodes: [layer_input_62], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg754_1, reinterpret_tensor(buf446, (128, 512), (512, 1), 0), reinterpret_tensor(arg753_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf447)
    del arg753_1
    del arg754_1
    buf448 = reinterpret_tensor(buf447, (1, 128, 128), (16384, 128, 1), 0); del buf447  # reuse
    cpp_fused_add_mul_146(c_void_p(buf448.data_ptr()), c_void_p(arg196_1.data_ptr()), c_void_p(arg197_1.data_ptr()))
    del arg196_1
    del arg197_1
    buf449 = buf443; del buf443  # reuse
    # Source Nodes: [mixed_query_layer_12], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg756_1, reinterpret_tensor(buf448, (128, 128), (128, 1), 0), reinterpret_tensor(arg755_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf449)
    del arg755_1
    del arg756_1
    buf450 = reinterpret_tensor(buf416, (128, 128), (128, 1), 0); del buf416  # reuse
    # Source Nodes: [mixed_key_layer_12], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg758_1, reinterpret_tensor(buf448, (128, 128), (128, 1), 0), reinterpret_tensor(arg757_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf450)
    del arg757_1
    del arg758_1
    buf451 = reinterpret_tensor(buf448, (128, 128), (128, 1), 0); del buf448  # reuse
    # Source Nodes: [mixed_value_layer_12], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg760_1, reinterpret_tensor(buf446, (128, 512), (512, 1), 0), reinterpret_tensor(arg759_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf451)
    del arg759_1
    del arg760_1
    buf452 = reinterpret_tensor(buf449, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf449  # reuse
    buf453 = reinterpret_tensor(buf450, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf450  # reuse
    buf454 = reinterpret_tensor(buf451, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf451  # reuse
    cpp_fused_147(c_void_p(buf452.data_ptr()), c_void_p(buf453.data_ptr()), c_void_p(buf454.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf455 = aten._scaled_dot_product_flash_attention(buf452, buf453, buf454, scale=0.17677669529663687)
    del buf452
    buf456 = buf455[0]
    del buf455
    buf463 = reinterpret_tensor(buf454, (128, 128), (128, 1), 0); del buf454  # reuse
    # Source Nodes: [layer_outputs_168], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg762_1, reinterpret_tensor(buf456, (128, 128), (128, 1), 0), reinterpret_tensor(arg761_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf463)
    del arg761_1
    del arg762_1
    buf464 = reinterpret_tensor(buf456, (128, 128), (128, 1), 0); del buf456  # reuse
    # Source Nodes: [layer_input_60], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg752_1, reinterpret_tensor(buf446, (128, 512), (512, 1), 0), reinterpret_tensor(arg751_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf464)
    del arg751_1
    del arg752_1
    buf465 = reinterpret_tensor(buf463, (1, 128, 128), (16384, 128, 1), 0); del buf463  # reuse
    cpp_fused_add_mul_148(c_void_p(buf465.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(arg194_1.data_ptr()), c_void_p(arg195_1.data_ptr()), c_void_p(arg198_1.data_ptr()), c_void_p(arg199_1.data_ptr()))
    del arg194_1
    del arg195_1
    del arg198_1
    del arg199_1
    buf466 = buf445; del buf445  # reuse
    # Source Nodes: [hidden_states_108], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg764_1, reinterpret_tensor(buf465, (128, 128), (128, 1), 0), reinterpret_tensor(arg763_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf466)
    del arg763_1
    del arg764_1
    buf467 = reinterpret_tensor(buf466, (1, 128, 512), (65536, 512, 1), 0); del buf466  # reuse
    cpp_fused_relu_149(c_void_p(buf467.data_ptr()))
    buf468 = buf464; del buf464  # reuse
    # Source Nodes: [layer_outputs_170], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg766_1, reinterpret_tensor(buf467, (128, 512), (512, 1), 0), reinterpret_tensor(arg765_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf468)
    del arg765_1
    del arg766_1
    buf469 = buf465; del buf465  # reuse
    cpp_fused_add_mul_150(c_void_p(buf469.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(arg200_1.data_ptr()), c_void_p(arg201_1.data_ptr()))
    del arg200_1
    del arg201_1
    buf470 = reinterpret_tensor(buf467, (128, 512), (512, 1), 0); del buf467  # reuse
    # Source Nodes: [hidden_states_110], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg768_1, reinterpret_tensor(buf469, (128, 128), (128, 1), 0), reinterpret_tensor(arg767_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf470)
    del arg767_1
    del arg768_1
    buf471 = reinterpret_tensor(buf470, (1, 128, 512), (65536, 512, 1), 0); del buf470  # reuse
    cpp_fused_relu_151(c_void_p(buf471.data_ptr()))
    buf472 = buf468; del buf468  # reuse
    # Source Nodes: [layer_outputs_173], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg770_1, reinterpret_tensor(buf471, (128, 512), (512, 1), 0), reinterpret_tensor(arg769_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf472)
    del arg769_1
    del arg770_1
    buf473 = buf469; del buf469  # reuse
    cpp_fused_add_mul_152(c_void_p(buf473.data_ptr()), c_void_p(buf472.data_ptr()), c_void_p(arg202_1.data_ptr()), c_void_p(arg203_1.data_ptr()))
    del arg202_1
    del arg203_1
    buf474 = reinterpret_tensor(buf471, (128, 512), (512, 1), 0); del buf471  # reuse
    # Source Nodes: [hidden_states_112], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg772_1, reinterpret_tensor(buf473, (128, 128), (128, 1), 0), reinterpret_tensor(arg771_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf474)
    del arg771_1
    del arg772_1
    buf475 = reinterpret_tensor(buf474, (1, 128, 512), (65536, 512, 1), 0); del buf474  # reuse
    cpp_fused_relu_153(c_void_p(buf475.data_ptr()))
    buf476 = buf472; del buf472  # reuse
    # Source Nodes: [layer_outputs_176], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg774_1, reinterpret_tensor(buf475, (128, 512), (512, 1), 0), reinterpret_tensor(arg773_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf476)
    del arg773_1
    del arg774_1
    buf477 = buf473; del buf473  # reuse
    cpp_fused_add_mul_154(c_void_p(buf477.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(arg204_1.data_ptr()), c_void_p(arg205_1.data_ptr()))
    del arg204_1
    del arg205_1
    buf478 = reinterpret_tensor(buf475, (128, 512), (512, 1), 0); del buf475  # reuse
    # Source Nodes: [hidden_states_114], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg776_1, reinterpret_tensor(buf477, (128, 128), (128, 1), 0), reinterpret_tensor(arg775_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf478)
    del arg775_1
    del arg776_1
    buf479 = reinterpret_tensor(buf478, (1, 128, 512), (65536, 512, 1), 0); del buf478  # reuse
    cpp_fused_relu_155(c_void_p(buf479.data_ptr()))
    buf480 = buf476; del buf476  # reuse
    # Source Nodes: [layer_output_48], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg778_1, reinterpret_tensor(buf479, (128, 512), (512, 1), 0), reinterpret_tensor(arg777_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf480)
    del arg777_1
    del arg778_1
    buf481 = buf477; del buf477  # reuse
    cpp_fused_add_mul_156(c_void_p(buf481.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(arg206_1.data_ptr()), c_void_p(arg207_1.data_ptr()))
    del arg206_1
    del arg207_1
    buf482 = reinterpret_tensor(buf479, (128, 512), (512, 1), 0); del buf479  # reuse
    # Source Nodes: [layer_outputs_179], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg780_1, reinterpret_tensor(buf481, (128, 128), (128, 1), 0), reinterpret_tensor(arg779_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf482)
    del arg779_1
    del arg780_1
    buf483 = buf446; del buf446  # reuse
    cpp_fused_add_mul_157(c_void_p(buf483.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(arg208_1.data_ptr()), c_void_p(arg209_1.data_ptr()))
    del arg208_1
    del arg209_1
    buf484 = reinterpret_tensor(buf481, (128, 128), (128, 1), 0); del buf481  # reuse
    # Source Nodes: [layer_input_67], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg784_1, reinterpret_tensor(buf483, (128, 512), (512, 1), 0), reinterpret_tensor(arg783_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf484)
    del arg783_1
    del arg784_1
    buf485 = reinterpret_tensor(buf484, (1, 128, 128), (16384, 128, 1), 0); del buf484  # reuse
    cpp_fused_add_mul_158(c_void_p(buf485.data_ptr()), c_void_p(arg212_1.data_ptr()), c_void_p(arg213_1.data_ptr()))
    del arg212_1
    del arg213_1
    buf486 = buf480; del buf480  # reuse
    # Source Nodes: [mixed_query_layer_13], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg786_1, reinterpret_tensor(buf485, (128, 128), (128, 1), 0), reinterpret_tensor(arg785_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf486)
    del arg785_1
    del arg786_1
    buf487 = reinterpret_tensor(buf453, (128, 128), (128, 1), 0); del buf453  # reuse
    # Source Nodes: [mixed_key_layer_13], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg788_1, reinterpret_tensor(buf485, (128, 128), (128, 1), 0), reinterpret_tensor(arg787_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf487)
    del arg787_1
    del arg788_1
    buf488 = reinterpret_tensor(buf485, (128, 128), (128, 1), 0); del buf485  # reuse
    # Source Nodes: [mixed_value_layer_13], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg790_1, reinterpret_tensor(buf483, (128, 512), (512, 1), 0), reinterpret_tensor(arg789_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf488)
    del arg789_1
    del arg790_1
    buf489 = reinterpret_tensor(buf486, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf486  # reuse
    buf490 = reinterpret_tensor(buf487, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf487  # reuse
    buf491 = reinterpret_tensor(buf488, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf488  # reuse
    cpp_fused_159(c_void_p(buf489.data_ptr()), c_void_p(buf490.data_ptr()), c_void_p(buf491.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf492 = aten._scaled_dot_product_flash_attention(buf489, buf490, buf491, scale=0.17677669529663687)
    del buf489
    buf493 = buf492[0]
    del buf492
    buf500 = reinterpret_tensor(buf491, (128, 128), (128, 1), 0); del buf491  # reuse
    # Source Nodes: [layer_outputs_182], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg792_1, reinterpret_tensor(buf493, (128, 128), (128, 1), 0), reinterpret_tensor(arg791_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf500)
    del arg791_1
    del arg792_1
    buf501 = reinterpret_tensor(buf493, (128, 128), (128, 1), 0); del buf493  # reuse
    # Source Nodes: [layer_input_65], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg782_1, reinterpret_tensor(buf483, (128, 512), (512, 1), 0), reinterpret_tensor(arg781_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf501)
    del arg781_1
    del arg782_1
    buf502 = reinterpret_tensor(buf500, (1, 128, 128), (16384, 128, 1), 0); del buf500  # reuse
    cpp_fused_add_mul_160(c_void_p(buf502.data_ptr()), c_void_p(buf501.data_ptr()), c_void_p(arg210_1.data_ptr()), c_void_p(arg211_1.data_ptr()), c_void_p(arg214_1.data_ptr()), c_void_p(arg215_1.data_ptr()))
    del arg210_1
    del arg211_1
    del arg214_1
    del arg215_1
    buf503 = buf482; del buf482  # reuse
    # Source Nodes: [hidden_states_117], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg794_1, reinterpret_tensor(buf502, (128, 128), (128, 1), 0), reinterpret_tensor(arg793_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf503)
    del arg793_1
    del arg794_1
    buf504 = reinterpret_tensor(buf503, (1, 128, 512), (65536, 512, 1), 0); del buf503  # reuse
    cpp_fused_relu_161(c_void_p(buf504.data_ptr()))
    buf505 = buf501; del buf501  # reuse
    # Source Nodes: [layer_outputs_184], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg796_1, reinterpret_tensor(buf504, (128, 512), (512, 1), 0), reinterpret_tensor(arg795_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf505)
    del arg795_1
    del arg796_1
    buf506 = buf502; del buf502  # reuse
    cpp_fused_add_mul_162(c_void_p(buf506.data_ptr()), c_void_p(buf505.data_ptr()), c_void_p(arg216_1.data_ptr()), c_void_p(arg217_1.data_ptr()))
    del arg216_1
    del arg217_1
    buf507 = reinterpret_tensor(buf504, (128, 512), (512, 1), 0); del buf504  # reuse
    # Source Nodes: [hidden_states_119], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg798_1, reinterpret_tensor(buf506, (128, 128), (128, 1), 0), reinterpret_tensor(arg797_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf507)
    del arg797_1
    del arg798_1
    buf508 = reinterpret_tensor(buf507, (1, 128, 512), (65536, 512, 1), 0); del buf507  # reuse
    cpp_fused_relu_163(c_void_p(buf508.data_ptr()))
    buf509 = buf505; del buf505  # reuse
    # Source Nodes: [layer_outputs_187], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg800_1, reinterpret_tensor(buf508, (128, 512), (512, 1), 0), reinterpret_tensor(arg799_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf509)
    del arg799_1
    del arg800_1
    buf510 = buf506; del buf506  # reuse
    cpp_fused_add_mul_164(c_void_p(buf510.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(arg218_1.data_ptr()), c_void_p(arg219_1.data_ptr()))
    del arg218_1
    del arg219_1
    buf511 = reinterpret_tensor(buf508, (128, 512), (512, 1), 0); del buf508  # reuse
    # Source Nodes: [hidden_states_121], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg802_1, reinterpret_tensor(buf510, (128, 128), (128, 1), 0), reinterpret_tensor(arg801_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf511)
    del arg801_1
    del arg802_1
    buf512 = reinterpret_tensor(buf511, (1, 128, 512), (65536, 512, 1), 0); del buf511  # reuse
    cpp_fused_relu_165(c_void_p(buf512.data_ptr()))
    buf513 = buf509; del buf509  # reuse
    # Source Nodes: [layer_outputs_190], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg804_1, reinterpret_tensor(buf512, (128, 512), (512, 1), 0), reinterpret_tensor(arg803_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf513)
    del arg803_1
    del arg804_1
    buf514 = buf510; del buf510  # reuse
    cpp_fused_add_mul_166(c_void_p(buf514.data_ptr()), c_void_p(buf513.data_ptr()), c_void_p(arg220_1.data_ptr()), c_void_p(arg221_1.data_ptr()))
    del arg220_1
    del arg221_1
    buf515 = reinterpret_tensor(buf512, (128, 512), (512, 1), 0); del buf512  # reuse
    # Source Nodes: [hidden_states_123], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg806_1, reinterpret_tensor(buf514, (128, 128), (128, 1), 0), reinterpret_tensor(arg805_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf515)
    del arg805_1
    del arg806_1
    buf516 = reinterpret_tensor(buf515, (1, 128, 512), (65536, 512, 1), 0); del buf515  # reuse
    cpp_fused_relu_167(c_void_p(buf516.data_ptr()))
    buf517 = buf513; del buf513  # reuse
    # Source Nodes: [layer_output_52], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg808_1, reinterpret_tensor(buf516, (128, 512), (512, 1), 0), reinterpret_tensor(arg807_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf517)
    del arg807_1
    del arg808_1
    buf518 = buf514; del buf514  # reuse
    cpp_fused_add_mul_168(c_void_p(buf518.data_ptr()), c_void_p(buf517.data_ptr()), c_void_p(arg222_1.data_ptr()), c_void_p(arg223_1.data_ptr()))
    del arg222_1
    del arg223_1
    buf519 = reinterpret_tensor(buf516, (128, 512), (512, 1), 0); del buf516  # reuse
    # Source Nodes: [layer_outputs_193], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg810_1, reinterpret_tensor(buf518, (128, 128), (128, 1), 0), reinterpret_tensor(arg809_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf519)
    del arg809_1
    del arg810_1
    buf520 = buf483; del buf483  # reuse
    cpp_fused_add_mul_169(c_void_p(buf520.data_ptr()), c_void_p(buf519.data_ptr()), c_void_p(arg224_1.data_ptr()), c_void_p(arg225_1.data_ptr()))
    del arg224_1
    del arg225_1
    buf521 = reinterpret_tensor(buf518, (128, 128), (128, 1), 0); del buf518  # reuse
    # Source Nodes: [layer_input_72], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg814_1, reinterpret_tensor(buf520, (128, 512), (512, 1), 0), reinterpret_tensor(arg813_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf521)
    del arg813_1
    del arg814_1
    buf522 = reinterpret_tensor(buf521, (1, 128, 128), (16384, 128, 1), 0); del buf521  # reuse
    cpp_fused_add_mul_170(c_void_p(buf522.data_ptr()), c_void_p(arg228_1.data_ptr()), c_void_p(arg229_1.data_ptr()))
    del arg228_1
    del arg229_1
    buf523 = buf517; del buf517  # reuse
    # Source Nodes: [mixed_query_layer_14], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg816_1, reinterpret_tensor(buf522, (128, 128), (128, 1), 0), reinterpret_tensor(arg815_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf523)
    del arg815_1
    del arg816_1
    buf524 = reinterpret_tensor(buf490, (128, 128), (128, 1), 0); del buf490  # reuse
    # Source Nodes: [mixed_key_layer_14], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg818_1, reinterpret_tensor(buf522, (128, 128), (128, 1), 0), reinterpret_tensor(arg817_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf524)
    del arg817_1
    del arg818_1
    buf525 = reinterpret_tensor(buf522, (128, 128), (128, 1), 0); del buf522  # reuse
    # Source Nodes: [mixed_value_layer_14], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg820_1, reinterpret_tensor(buf520, (128, 512), (512, 1), 0), reinterpret_tensor(arg819_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf525)
    del arg819_1
    del arg820_1
    buf526 = reinterpret_tensor(buf523, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf523  # reuse
    buf527 = reinterpret_tensor(buf524, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf524  # reuse
    buf528 = reinterpret_tensor(buf525, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf525  # reuse
    cpp_fused_171(c_void_p(buf526.data_ptr()), c_void_p(buf527.data_ptr()), c_void_p(buf528.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf529 = aten._scaled_dot_product_flash_attention(buf526, buf527, buf528, scale=0.17677669529663687)
    del buf526
    buf530 = buf529[0]
    del buf529
    buf537 = reinterpret_tensor(buf528, (128, 128), (128, 1), 0); del buf528  # reuse
    # Source Nodes: [layer_outputs_196], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg822_1, reinterpret_tensor(buf530, (128, 128), (128, 1), 0), reinterpret_tensor(arg821_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf537)
    del arg821_1
    del arg822_1
    buf538 = reinterpret_tensor(buf530, (128, 128), (128, 1), 0); del buf530  # reuse
    # Source Nodes: [layer_input_70], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg812_1, reinterpret_tensor(buf520, (128, 512), (512, 1), 0), reinterpret_tensor(arg811_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf538)
    del arg811_1
    del arg812_1
    buf539 = reinterpret_tensor(buf537, (1, 128, 128), (16384, 128, 1), 0); del buf537  # reuse
    cpp_fused_add_mul_172(c_void_p(buf539.data_ptr()), c_void_p(buf538.data_ptr()), c_void_p(arg226_1.data_ptr()), c_void_p(arg227_1.data_ptr()), c_void_p(arg230_1.data_ptr()), c_void_p(arg231_1.data_ptr()))
    del arg226_1
    del arg227_1
    del arg230_1
    del arg231_1
    buf540 = buf519; del buf519  # reuse
    # Source Nodes: [hidden_states_126], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg824_1, reinterpret_tensor(buf539, (128, 128), (128, 1), 0), reinterpret_tensor(arg823_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf540)
    del arg823_1
    del arg824_1
    buf541 = reinterpret_tensor(buf540, (1, 128, 512), (65536, 512, 1), 0); del buf540  # reuse
    cpp_fused_relu_173(c_void_p(buf541.data_ptr()))
    buf542 = buf538; del buf538  # reuse
    # Source Nodes: [layer_outputs_198], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg826_1, reinterpret_tensor(buf541, (128, 512), (512, 1), 0), reinterpret_tensor(arg825_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf542)
    del arg825_1
    del arg826_1
    buf543 = buf539; del buf539  # reuse
    cpp_fused_add_mul_174(c_void_p(buf543.data_ptr()), c_void_p(buf542.data_ptr()), c_void_p(arg232_1.data_ptr()), c_void_p(arg233_1.data_ptr()))
    del arg232_1
    del arg233_1
    buf544 = reinterpret_tensor(buf541, (128, 512), (512, 1), 0); del buf541  # reuse
    # Source Nodes: [hidden_states_128], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg828_1, reinterpret_tensor(buf543, (128, 128), (128, 1), 0), reinterpret_tensor(arg827_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf544)
    del arg827_1
    del arg828_1
    buf545 = reinterpret_tensor(buf544, (1, 128, 512), (65536, 512, 1), 0); del buf544  # reuse
    cpp_fused_relu_175(c_void_p(buf545.data_ptr()))
    buf546 = buf542; del buf542  # reuse
    # Source Nodes: [layer_outputs_201], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg830_1, reinterpret_tensor(buf545, (128, 512), (512, 1), 0), reinterpret_tensor(arg829_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf546)
    del arg829_1
    del arg830_1
    buf547 = buf543; del buf543  # reuse
    cpp_fused_add_mul_176(c_void_p(buf547.data_ptr()), c_void_p(buf546.data_ptr()), c_void_p(arg234_1.data_ptr()), c_void_p(arg235_1.data_ptr()))
    del arg234_1
    del arg235_1
    buf548 = reinterpret_tensor(buf545, (128, 512), (512, 1), 0); del buf545  # reuse
    # Source Nodes: [hidden_states_130], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg832_1, reinterpret_tensor(buf547, (128, 128), (128, 1), 0), reinterpret_tensor(arg831_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf548)
    del arg831_1
    del arg832_1
    buf549 = reinterpret_tensor(buf548, (1, 128, 512), (65536, 512, 1), 0); del buf548  # reuse
    cpp_fused_relu_177(c_void_p(buf549.data_ptr()))
    buf550 = buf546; del buf546  # reuse
    # Source Nodes: [layer_outputs_204], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg834_1, reinterpret_tensor(buf549, (128, 512), (512, 1), 0), reinterpret_tensor(arg833_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf550)
    del arg833_1
    del arg834_1
    buf551 = buf547; del buf547  # reuse
    cpp_fused_add_mul_178(c_void_p(buf551.data_ptr()), c_void_p(buf550.data_ptr()), c_void_p(arg236_1.data_ptr()), c_void_p(arg237_1.data_ptr()))
    del arg236_1
    del arg237_1
    buf552 = reinterpret_tensor(buf549, (128, 512), (512, 1), 0); del buf549  # reuse
    # Source Nodes: [hidden_states_132], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg836_1, reinterpret_tensor(buf551, (128, 128), (128, 1), 0), reinterpret_tensor(arg835_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf552)
    del arg835_1
    del arg836_1
    buf553 = reinterpret_tensor(buf552, (1, 128, 512), (65536, 512, 1), 0); del buf552  # reuse
    cpp_fused_relu_179(c_void_p(buf553.data_ptr()))
    buf554 = buf550; del buf550  # reuse
    # Source Nodes: [layer_output_56], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg838_1, reinterpret_tensor(buf553, (128, 512), (512, 1), 0), reinterpret_tensor(arg837_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf554)
    del arg837_1
    del arg838_1
    buf555 = buf551; del buf551  # reuse
    cpp_fused_add_mul_180(c_void_p(buf555.data_ptr()), c_void_p(buf554.data_ptr()), c_void_p(arg238_1.data_ptr()), c_void_p(arg239_1.data_ptr()))
    del arg238_1
    del arg239_1
    buf556 = reinterpret_tensor(buf553, (128, 512), (512, 1), 0); del buf553  # reuse
    # Source Nodes: [layer_outputs_207], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg840_1, reinterpret_tensor(buf555, (128, 128), (128, 1), 0), reinterpret_tensor(arg839_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf556)
    del arg839_1
    del arg840_1
    buf557 = buf520; del buf520  # reuse
    cpp_fused_add_mul_181(c_void_p(buf557.data_ptr()), c_void_p(buf556.data_ptr()), c_void_p(arg240_1.data_ptr()), c_void_p(arg241_1.data_ptr()))
    del arg240_1
    del arg241_1
    buf558 = reinterpret_tensor(buf555, (128, 128), (128, 1), 0); del buf555  # reuse
    # Source Nodes: [layer_input_77], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg844_1, reinterpret_tensor(buf557, (128, 512), (512, 1), 0), reinterpret_tensor(arg843_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf558)
    del arg843_1
    del arg844_1
    buf559 = reinterpret_tensor(buf558, (1, 128, 128), (16384, 128, 1), 0); del buf558  # reuse
    cpp_fused_add_mul_182(c_void_p(buf559.data_ptr()), c_void_p(arg244_1.data_ptr()), c_void_p(arg245_1.data_ptr()))
    del arg244_1
    del arg245_1
    buf560 = buf554; del buf554  # reuse
    # Source Nodes: [mixed_query_layer_15], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg846_1, reinterpret_tensor(buf559, (128, 128), (128, 1), 0), reinterpret_tensor(arg845_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf560)
    del arg845_1
    del arg846_1
    buf561 = reinterpret_tensor(buf527, (128, 128), (128, 1), 0); del buf527  # reuse
    # Source Nodes: [mixed_key_layer_15], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg848_1, reinterpret_tensor(buf559, (128, 128), (128, 1), 0), reinterpret_tensor(arg847_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf561)
    del arg847_1
    del arg848_1
    buf562 = reinterpret_tensor(buf559, (128, 128), (128, 1), 0); del buf559  # reuse
    # Source Nodes: [mixed_value_layer_15], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg850_1, reinterpret_tensor(buf557, (128, 512), (512, 1), 0), reinterpret_tensor(arg849_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf562)
    del arg849_1
    del arg850_1
    buf563 = reinterpret_tensor(buf560, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf560  # reuse
    buf564 = reinterpret_tensor(buf561, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf561  # reuse
    buf565 = reinterpret_tensor(buf562, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf562  # reuse
    cpp_fused_183(c_void_p(buf563.data_ptr()), c_void_p(buf564.data_ptr()), c_void_p(buf565.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf566 = aten._scaled_dot_product_flash_attention(buf563, buf564, buf565, scale=0.17677669529663687)
    del buf563
    buf567 = buf566[0]
    del buf566
    buf574 = reinterpret_tensor(buf565, (128, 128), (128, 1), 0); del buf565  # reuse
    # Source Nodes: [layer_outputs_210], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg852_1, reinterpret_tensor(buf567, (128, 128), (128, 1), 0), reinterpret_tensor(arg851_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf574)
    del arg851_1
    del arg852_1
    buf575 = reinterpret_tensor(buf567, (128, 128), (128, 1), 0); del buf567  # reuse
    # Source Nodes: [layer_input_75], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg842_1, reinterpret_tensor(buf557, (128, 512), (512, 1), 0), reinterpret_tensor(arg841_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf575)
    del arg841_1
    del arg842_1
    buf576 = reinterpret_tensor(buf574, (1, 128, 128), (16384, 128, 1), 0); del buf574  # reuse
    cpp_fused_add_mul_184(c_void_p(buf576.data_ptr()), c_void_p(buf575.data_ptr()), c_void_p(arg242_1.data_ptr()), c_void_p(arg243_1.data_ptr()), c_void_p(arg246_1.data_ptr()), c_void_p(arg247_1.data_ptr()))
    del arg242_1
    del arg243_1
    del arg246_1
    del arg247_1
    buf577 = buf556; del buf556  # reuse
    # Source Nodes: [hidden_states_135], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg854_1, reinterpret_tensor(buf576, (128, 128), (128, 1), 0), reinterpret_tensor(arg853_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf577)
    del arg853_1
    del arg854_1
    buf578 = reinterpret_tensor(buf577, (1, 128, 512), (65536, 512, 1), 0); del buf577  # reuse
    cpp_fused_relu_185(c_void_p(buf578.data_ptr()))
    buf579 = buf575; del buf575  # reuse
    # Source Nodes: [layer_outputs_212], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg856_1, reinterpret_tensor(buf578, (128, 512), (512, 1), 0), reinterpret_tensor(arg855_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf579)
    del arg855_1
    del arg856_1
    buf580 = buf576; del buf576  # reuse
    cpp_fused_add_mul_186(c_void_p(buf580.data_ptr()), c_void_p(buf579.data_ptr()), c_void_p(arg248_1.data_ptr()), c_void_p(arg249_1.data_ptr()))
    del arg248_1
    del arg249_1
    buf581 = reinterpret_tensor(buf578, (128, 512), (512, 1), 0); del buf578  # reuse
    # Source Nodes: [hidden_states_137], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg858_1, reinterpret_tensor(buf580, (128, 128), (128, 1), 0), reinterpret_tensor(arg857_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf581)
    del arg857_1
    del arg858_1
    buf582 = reinterpret_tensor(buf581, (1, 128, 512), (65536, 512, 1), 0); del buf581  # reuse
    cpp_fused_relu_187(c_void_p(buf582.data_ptr()))
    buf583 = buf579; del buf579  # reuse
    # Source Nodes: [layer_outputs_215], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg860_1, reinterpret_tensor(buf582, (128, 512), (512, 1), 0), reinterpret_tensor(arg859_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf583)
    del arg859_1
    del arg860_1
    buf584 = buf580; del buf580  # reuse
    cpp_fused_add_mul_188(c_void_p(buf584.data_ptr()), c_void_p(buf583.data_ptr()), c_void_p(arg250_1.data_ptr()), c_void_p(arg251_1.data_ptr()))
    del arg250_1
    del arg251_1
    buf585 = reinterpret_tensor(buf582, (128, 512), (512, 1), 0); del buf582  # reuse
    # Source Nodes: [hidden_states_139], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg862_1, reinterpret_tensor(buf584, (128, 128), (128, 1), 0), reinterpret_tensor(arg861_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf585)
    del arg861_1
    del arg862_1
    buf586 = reinterpret_tensor(buf585, (1, 128, 512), (65536, 512, 1), 0); del buf585  # reuse
    cpp_fused_relu_189(c_void_p(buf586.data_ptr()))
    buf587 = buf583; del buf583  # reuse
    # Source Nodes: [layer_outputs_218], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg864_1, reinterpret_tensor(buf586, (128, 512), (512, 1), 0), reinterpret_tensor(arg863_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf587)
    del arg863_1
    del arg864_1
    buf588 = buf584; del buf584  # reuse
    cpp_fused_add_mul_190(c_void_p(buf588.data_ptr()), c_void_p(buf587.data_ptr()), c_void_p(arg252_1.data_ptr()), c_void_p(arg253_1.data_ptr()))
    del arg252_1
    del arg253_1
    buf589 = reinterpret_tensor(buf586, (128, 512), (512, 1), 0); del buf586  # reuse
    # Source Nodes: [hidden_states_141], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg866_1, reinterpret_tensor(buf588, (128, 128), (128, 1), 0), reinterpret_tensor(arg865_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf589)
    del arg865_1
    del arg866_1
    buf590 = reinterpret_tensor(buf589, (1, 128, 512), (65536, 512, 1), 0); del buf589  # reuse
    cpp_fused_relu_191(c_void_p(buf590.data_ptr()))
    buf591 = buf587; del buf587  # reuse
    # Source Nodes: [layer_output_60], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg868_1, reinterpret_tensor(buf590, (128, 512), (512, 1), 0), reinterpret_tensor(arg867_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf591)
    del arg867_1
    del arg868_1
    buf592 = buf588; del buf588  # reuse
    cpp_fused_add_mul_192(c_void_p(buf592.data_ptr()), c_void_p(buf591.data_ptr()), c_void_p(arg254_1.data_ptr()), c_void_p(arg255_1.data_ptr()))
    del arg254_1
    del arg255_1
    buf593 = reinterpret_tensor(buf590, (128, 512), (512, 1), 0); del buf590  # reuse
    # Source Nodes: [layer_outputs_221], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg870_1, reinterpret_tensor(buf592, (128, 128), (128, 1), 0), reinterpret_tensor(arg869_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf593)
    del arg869_1
    del arg870_1
    buf594 = buf557; del buf557  # reuse
    cpp_fused_add_mul_193(c_void_p(buf594.data_ptr()), c_void_p(buf593.data_ptr()), c_void_p(arg256_1.data_ptr()), c_void_p(arg257_1.data_ptr()))
    del arg256_1
    del arg257_1
    buf595 = reinterpret_tensor(buf592, (128, 128), (128, 1), 0); del buf592  # reuse
    # Source Nodes: [layer_input_82], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg874_1, reinterpret_tensor(buf594, (128, 512), (512, 1), 0), reinterpret_tensor(arg873_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf595)
    del arg873_1
    del arg874_1
    buf596 = reinterpret_tensor(buf595, (1, 128, 128), (16384, 128, 1), 0); del buf595  # reuse
    cpp_fused_add_mul_194(c_void_p(buf596.data_ptr()), c_void_p(arg260_1.data_ptr()), c_void_p(arg261_1.data_ptr()))
    del arg260_1
    del arg261_1
    buf597 = buf591; del buf591  # reuse
    # Source Nodes: [mixed_query_layer_16], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg876_1, reinterpret_tensor(buf596, (128, 128), (128, 1), 0), reinterpret_tensor(arg875_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf597)
    del arg875_1
    del arg876_1
    buf598 = reinterpret_tensor(buf564, (128, 128), (128, 1), 0); del buf564  # reuse
    # Source Nodes: [mixed_key_layer_16], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg878_1, reinterpret_tensor(buf596, (128, 128), (128, 1), 0), reinterpret_tensor(arg877_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf598)
    del arg877_1
    del arg878_1
    buf599 = reinterpret_tensor(buf596, (128, 128), (128, 1), 0); del buf596  # reuse
    # Source Nodes: [mixed_value_layer_16], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg880_1, reinterpret_tensor(buf594, (128, 512), (512, 1), 0), reinterpret_tensor(arg879_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf599)
    del arg879_1
    del arg880_1
    buf600 = reinterpret_tensor(buf597, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf597  # reuse
    buf601 = reinterpret_tensor(buf598, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf598  # reuse
    buf602 = reinterpret_tensor(buf599, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf599  # reuse
    cpp_fused_195(c_void_p(buf600.data_ptr()), c_void_p(buf601.data_ptr()), c_void_p(buf602.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf603 = aten._scaled_dot_product_flash_attention(buf600, buf601, buf602, scale=0.17677669529663687)
    del buf600
    buf604 = buf603[0]
    del buf603
    buf611 = reinterpret_tensor(buf602, (128, 128), (128, 1), 0); del buf602  # reuse
    # Source Nodes: [layer_outputs_224], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg882_1, reinterpret_tensor(buf604, (128, 128), (128, 1), 0), reinterpret_tensor(arg881_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf611)
    del arg881_1
    del arg882_1
    buf612 = reinterpret_tensor(buf604, (128, 128), (128, 1), 0); del buf604  # reuse
    # Source Nodes: [layer_input_80], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg872_1, reinterpret_tensor(buf594, (128, 512), (512, 1), 0), reinterpret_tensor(arg871_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf612)
    del arg871_1
    del arg872_1
    buf613 = reinterpret_tensor(buf611, (1, 128, 128), (16384, 128, 1), 0); del buf611  # reuse
    cpp_fused_add_mul_196(c_void_p(buf613.data_ptr()), c_void_p(buf612.data_ptr()), c_void_p(arg258_1.data_ptr()), c_void_p(arg259_1.data_ptr()), c_void_p(arg262_1.data_ptr()), c_void_p(arg263_1.data_ptr()))
    del arg258_1
    del arg259_1
    del arg262_1
    del arg263_1
    buf614 = buf593; del buf593  # reuse
    # Source Nodes: [hidden_states_144], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg884_1, reinterpret_tensor(buf613, (128, 128), (128, 1), 0), reinterpret_tensor(arg883_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf614)
    del arg883_1
    del arg884_1
    buf615 = reinterpret_tensor(buf614, (1, 128, 512), (65536, 512, 1), 0); del buf614  # reuse
    cpp_fused_relu_197(c_void_p(buf615.data_ptr()))
    buf616 = buf612; del buf612  # reuse
    # Source Nodes: [layer_outputs_226], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg886_1, reinterpret_tensor(buf615, (128, 512), (512, 1), 0), reinterpret_tensor(arg885_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf616)
    del arg885_1
    del arg886_1
    buf617 = buf613; del buf613  # reuse
    cpp_fused_add_mul_198(c_void_p(buf617.data_ptr()), c_void_p(buf616.data_ptr()), c_void_p(arg264_1.data_ptr()), c_void_p(arg265_1.data_ptr()))
    del arg264_1
    del arg265_1
    buf618 = reinterpret_tensor(buf615, (128, 512), (512, 1), 0); del buf615  # reuse
    # Source Nodes: [hidden_states_146], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg888_1, reinterpret_tensor(buf617, (128, 128), (128, 1), 0), reinterpret_tensor(arg887_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf618)
    del arg887_1
    del arg888_1
    buf619 = reinterpret_tensor(buf618, (1, 128, 512), (65536, 512, 1), 0); del buf618  # reuse
    cpp_fused_relu_199(c_void_p(buf619.data_ptr()))
    buf620 = buf616; del buf616  # reuse
    # Source Nodes: [layer_outputs_229], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg890_1, reinterpret_tensor(buf619, (128, 512), (512, 1), 0), reinterpret_tensor(arg889_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf620)
    del arg889_1
    del arg890_1
    buf621 = buf617; del buf617  # reuse
    cpp_fused_add_mul_200(c_void_p(buf621.data_ptr()), c_void_p(buf620.data_ptr()), c_void_p(arg266_1.data_ptr()), c_void_p(arg267_1.data_ptr()))
    del arg266_1
    del arg267_1
    buf622 = reinterpret_tensor(buf619, (128, 512), (512, 1), 0); del buf619  # reuse
    # Source Nodes: [hidden_states_148], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg892_1, reinterpret_tensor(buf621, (128, 128), (128, 1), 0), reinterpret_tensor(arg891_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf622)
    del arg891_1
    del arg892_1
    buf623 = reinterpret_tensor(buf622, (1, 128, 512), (65536, 512, 1), 0); del buf622  # reuse
    cpp_fused_relu_201(c_void_p(buf623.data_ptr()))
    buf624 = buf620; del buf620  # reuse
    # Source Nodes: [layer_outputs_232], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg894_1, reinterpret_tensor(buf623, (128, 512), (512, 1), 0), reinterpret_tensor(arg893_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf624)
    del arg893_1
    del arg894_1
    buf625 = buf621; del buf621  # reuse
    cpp_fused_add_mul_202(c_void_p(buf625.data_ptr()), c_void_p(buf624.data_ptr()), c_void_p(arg268_1.data_ptr()), c_void_p(arg269_1.data_ptr()))
    del arg268_1
    del arg269_1
    buf626 = reinterpret_tensor(buf623, (128, 512), (512, 1), 0); del buf623  # reuse
    # Source Nodes: [hidden_states_150], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg896_1, reinterpret_tensor(buf625, (128, 128), (128, 1), 0), reinterpret_tensor(arg895_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf626)
    del arg895_1
    del arg896_1
    buf627 = reinterpret_tensor(buf626, (1, 128, 512), (65536, 512, 1), 0); del buf626  # reuse
    cpp_fused_relu_203(c_void_p(buf627.data_ptr()))
    buf628 = buf624; del buf624  # reuse
    # Source Nodes: [layer_output_64], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg898_1, reinterpret_tensor(buf627, (128, 512), (512, 1), 0), reinterpret_tensor(arg897_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf628)
    del arg897_1
    del arg898_1
    buf629 = buf625; del buf625  # reuse
    cpp_fused_add_mul_204(c_void_p(buf629.data_ptr()), c_void_p(buf628.data_ptr()), c_void_p(arg270_1.data_ptr()), c_void_p(arg271_1.data_ptr()))
    del arg270_1
    del arg271_1
    buf630 = reinterpret_tensor(buf627, (128, 512), (512, 1), 0); del buf627  # reuse
    # Source Nodes: [layer_outputs_235], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg900_1, reinterpret_tensor(buf629, (128, 128), (128, 1), 0), reinterpret_tensor(arg899_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf630)
    del arg899_1
    del arg900_1
    buf631 = buf594; del buf594  # reuse
    cpp_fused_add_mul_205(c_void_p(buf631.data_ptr()), c_void_p(buf630.data_ptr()), c_void_p(arg272_1.data_ptr()), c_void_p(arg273_1.data_ptr()))
    del arg272_1
    del arg273_1
    buf632 = reinterpret_tensor(buf629, (128, 128), (128, 1), 0); del buf629  # reuse
    # Source Nodes: [layer_input_87], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg904_1, reinterpret_tensor(buf631, (128, 512), (512, 1), 0), reinterpret_tensor(arg903_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf632)
    del arg903_1
    del arg904_1
    buf633 = reinterpret_tensor(buf632, (1, 128, 128), (16384, 128, 1), 0); del buf632  # reuse
    cpp_fused_add_mul_206(c_void_p(buf633.data_ptr()), c_void_p(arg276_1.data_ptr()), c_void_p(arg277_1.data_ptr()))
    del arg276_1
    del arg277_1
    buf634 = buf628; del buf628  # reuse
    # Source Nodes: [mixed_query_layer_17], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg906_1, reinterpret_tensor(buf633, (128, 128), (128, 1), 0), reinterpret_tensor(arg905_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf634)
    del arg905_1
    del arg906_1
    buf635 = reinterpret_tensor(buf601, (128, 128), (128, 1), 0); del buf601  # reuse
    # Source Nodes: [mixed_key_layer_17], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg908_1, reinterpret_tensor(buf633, (128, 128), (128, 1), 0), reinterpret_tensor(arg907_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf635)
    del arg907_1
    del arg908_1
    buf636 = reinterpret_tensor(buf633, (128, 128), (128, 1), 0); del buf633  # reuse
    # Source Nodes: [mixed_value_layer_17], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg910_1, reinterpret_tensor(buf631, (128, 512), (512, 1), 0), reinterpret_tensor(arg909_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf636)
    del arg909_1
    del arg910_1
    buf637 = reinterpret_tensor(buf634, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf634  # reuse
    buf638 = reinterpret_tensor(buf635, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf635  # reuse
    buf639 = reinterpret_tensor(buf636, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf636  # reuse
    cpp_fused_207(c_void_p(buf637.data_ptr()), c_void_p(buf638.data_ptr()), c_void_p(buf639.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf640 = aten._scaled_dot_product_flash_attention(buf637, buf638, buf639, scale=0.17677669529663687)
    del buf637
    buf641 = buf640[0]
    del buf640
    buf648 = reinterpret_tensor(buf639, (128, 128), (128, 1), 0); del buf639  # reuse
    # Source Nodes: [layer_outputs_238], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg912_1, reinterpret_tensor(buf641, (128, 128), (128, 1), 0), reinterpret_tensor(arg911_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf648)
    del arg911_1
    del arg912_1
    buf649 = reinterpret_tensor(buf641, (128, 128), (128, 1), 0); del buf641  # reuse
    # Source Nodes: [layer_input_85], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg902_1, reinterpret_tensor(buf631, (128, 512), (512, 1), 0), reinterpret_tensor(arg901_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf649)
    del arg901_1
    del arg902_1
    buf650 = reinterpret_tensor(buf648, (1, 128, 128), (16384, 128, 1), 0); del buf648  # reuse
    cpp_fused_add_mul_208(c_void_p(buf650.data_ptr()), c_void_p(buf649.data_ptr()), c_void_p(arg274_1.data_ptr()), c_void_p(arg275_1.data_ptr()), c_void_p(arg278_1.data_ptr()), c_void_p(arg279_1.data_ptr()))
    del arg274_1
    del arg275_1
    del arg278_1
    del arg279_1
    buf651 = buf630; del buf630  # reuse
    # Source Nodes: [hidden_states_153], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg914_1, reinterpret_tensor(buf650, (128, 128), (128, 1), 0), reinterpret_tensor(arg913_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf651)
    del arg913_1
    del arg914_1
    buf652 = reinterpret_tensor(buf651, (1, 128, 512), (65536, 512, 1), 0); del buf651  # reuse
    cpp_fused_relu_209(c_void_p(buf652.data_ptr()))
    buf653 = buf649; del buf649  # reuse
    # Source Nodes: [layer_outputs_240], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg916_1, reinterpret_tensor(buf652, (128, 512), (512, 1), 0), reinterpret_tensor(arg915_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf653)
    del arg915_1
    del arg916_1
    buf654 = buf650; del buf650  # reuse
    cpp_fused_add_mul_210(c_void_p(buf654.data_ptr()), c_void_p(buf653.data_ptr()), c_void_p(arg280_1.data_ptr()), c_void_p(arg281_1.data_ptr()))
    del arg280_1
    del arg281_1
    buf655 = reinterpret_tensor(buf652, (128, 512), (512, 1), 0); del buf652  # reuse
    # Source Nodes: [hidden_states_155], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg918_1, reinterpret_tensor(buf654, (128, 128), (128, 1), 0), reinterpret_tensor(arg917_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf655)
    del arg917_1
    del arg918_1
    buf656 = reinterpret_tensor(buf655, (1, 128, 512), (65536, 512, 1), 0); del buf655  # reuse
    cpp_fused_relu_211(c_void_p(buf656.data_ptr()))
    buf657 = buf653; del buf653  # reuse
    # Source Nodes: [layer_outputs_243], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg920_1, reinterpret_tensor(buf656, (128, 512), (512, 1), 0), reinterpret_tensor(arg919_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf657)
    del arg919_1
    del arg920_1
    buf658 = buf654; del buf654  # reuse
    cpp_fused_add_mul_212(c_void_p(buf658.data_ptr()), c_void_p(buf657.data_ptr()), c_void_p(arg282_1.data_ptr()), c_void_p(arg283_1.data_ptr()))
    del arg282_1
    del arg283_1
    buf659 = reinterpret_tensor(buf656, (128, 512), (512, 1), 0); del buf656  # reuse
    # Source Nodes: [hidden_states_157], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg922_1, reinterpret_tensor(buf658, (128, 128), (128, 1), 0), reinterpret_tensor(arg921_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf659)
    del arg921_1
    del arg922_1
    buf660 = reinterpret_tensor(buf659, (1, 128, 512), (65536, 512, 1), 0); del buf659  # reuse
    cpp_fused_relu_213(c_void_p(buf660.data_ptr()))
    buf661 = buf657; del buf657  # reuse
    # Source Nodes: [layer_outputs_246], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg924_1, reinterpret_tensor(buf660, (128, 512), (512, 1), 0), reinterpret_tensor(arg923_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf661)
    del arg923_1
    del arg924_1
    buf662 = buf658; del buf658  # reuse
    cpp_fused_add_mul_214(c_void_p(buf662.data_ptr()), c_void_p(buf661.data_ptr()), c_void_p(arg284_1.data_ptr()), c_void_p(arg285_1.data_ptr()))
    del arg284_1
    del arg285_1
    buf663 = reinterpret_tensor(buf660, (128, 512), (512, 1), 0); del buf660  # reuse
    # Source Nodes: [hidden_states_159], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg926_1, reinterpret_tensor(buf662, (128, 128), (128, 1), 0), reinterpret_tensor(arg925_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf663)
    del arg925_1
    del arg926_1
    buf664 = reinterpret_tensor(buf663, (1, 128, 512), (65536, 512, 1), 0); del buf663  # reuse
    cpp_fused_relu_215(c_void_p(buf664.data_ptr()))
    buf665 = buf661; del buf661  # reuse
    # Source Nodes: [layer_output_68], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg928_1, reinterpret_tensor(buf664, (128, 512), (512, 1), 0), reinterpret_tensor(arg927_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf665)
    del arg927_1
    del arg928_1
    buf666 = buf662; del buf662  # reuse
    cpp_fused_add_mul_216(c_void_p(buf666.data_ptr()), c_void_p(buf665.data_ptr()), c_void_p(arg286_1.data_ptr()), c_void_p(arg287_1.data_ptr()))
    del arg286_1
    del arg287_1
    buf667 = reinterpret_tensor(buf664, (128, 512), (512, 1), 0); del buf664  # reuse
    # Source Nodes: [layer_outputs_249], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg930_1, reinterpret_tensor(buf666, (128, 128), (128, 1), 0), reinterpret_tensor(arg929_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf667)
    del arg929_1
    del arg930_1
    buf668 = buf631; del buf631  # reuse
    cpp_fused_add_mul_217(c_void_p(buf668.data_ptr()), c_void_p(buf667.data_ptr()), c_void_p(arg288_1.data_ptr()), c_void_p(arg289_1.data_ptr()))
    del arg288_1
    del arg289_1
    buf669 = reinterpret_tensor(buf666, (128, 128), (128, 1), 0); del buf666  # reuse
    # Source Nodes: [layer_input_92], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg934_1, reinterpret_tensor(buf668, (128, 512), (512, 1), 0), reinterpret_tensor(arg933_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf669)
    del arg933_1
    del arg934_1
    buf670 = reinterpret_tensor(buf669, (1, 128, 128), (16384, 128, 1), 0); del buf669  # reuse
    cpp_fused_add_mul_218(c_void_p(buf670.data_ptr()), c_void_p(arg292_1.data_ptr()), c_void_p(arg293_1.data_ptr()))
    del arg292_1
    del arg293_1
    buf671 = buf665; del buf665  # reuse
    # Source Nodes: [mixed_query_layer_18], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg936_1, reinterpret_tensor(buf670, (128, 128), (128, 1), 0), reinterpret_tensor(arg935_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf671)
    del arg935_1
    del arg936_1
    buf672 = reinterpret_tensor(buf638, (128, 128), (128, 1), 0); del buf638  # reuse
    # Source Nodes: [mixed_key_layer_18], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg938_1, reinterpret_tensor(buf670, (128, 128), (128, 1), 0), reinterpret_tensor(arg937_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf672)
    del arg937_1
    del arg938_1
    buf673 = reinterpret_tensor(buf670, (128, 128), (128, 1), 0); del buf670  # reuse
    # Source Nodes: [mixed_value_layer_18], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg940_1, reinterpret_tensor(buf668, (128, 512), (512, 1), 0), reinterpret_tensor(arg939_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf673)
    del arg939_1
    del arg940_1
    buf674 = reinterpret_tensor(buf671, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf671  # reuse
    buf675 = reinterpret_tensor(buf672, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf672  # reuse
    buf676 = reinterpret_tensor(buf673, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf673  # reuse
    cpp_fused_219(c_void_p(buf674.data_ptr()), c_void_p(buf675.data_ptr()), c_void_p(buf676.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf677 = aten._scaled_dot_product_flash_attention(buf674, buf675, buf676, scale=0.17677669529663687)
    del buf674
    buf678 = buf677[0]
    del buf677
    buf685 = reinterpret_tensor(buf676, (128, 128), (128, 1), 0); del buf676  # reuse
    # Source Nodes: [layer_outputs_252], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg942_1, reinterpret_tensor(buf678, (128, 128), (128, 1), 0), reinterpret_tensor(arg941_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf685)
    del arg941_1
    del arg942_1
    buf686 = reinterpret_tensor(buf678, (128, 128), (128, 1), 0); del buf678  # reuse
    # Source Nodes: [layer_input_90], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg932_1, reinterpret_tensor(buf668, (128, 512), (512, 1), 0), reinterpret_tensor(arg931_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf686)
    del arg931_1
    del arg932_1
    buf687 = reinterpret_tensor(buf685, (1, 128, 128), (16384, 128, 1), 0); del buf685  # reuse
    cpp_fused_add_mul_220(c_void_p(buf687.data_ptr()), c_void_p(buf686.data_ptr()), c_void_p(arg290_1.data_ptr()), c_void_p(arg291_1.data_ptr()), c_void_p(arg294_1.data_ptr()), c_void_p(arg295_1.data_ptr()))
    del arg290_1
    del arg291_1
    del arg294_1
    del arg295_1
    buf688 = buf667; del buf667  # reuse
    # Source Nodes: [hidden_states_162], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg944_1, reinterpret_tensor(buf687, (128, 128), (128, 1), 0), reinterpret_tensor(arg943_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf688)
    del arg943_1
    del arg944_1
    buf689 = reinterpret_tensor(buf688, (1, 128, 512), (65536, 512, 1), 0); del buf688  # reuse
    cpp_fused_relu_221(c_void_p(buf689.data_ptr()))
    buf690 = buf686; del buf686  # reuse
    # Source Nodes: [layer_outputs_254], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg946_1, reinterpret_tensor(buf689, (128, 512), (512, 1), 0), reinterpret_tensor(arg945_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf690)
    del arg945_1
    del arg946_1
    buf691 = buf687; del buf687  # reuse
    cpp_fused_add_mul_222(c_void_p(buf691.data_ptr()), c_void_p(buf690.data_ptr()), c_void_p(arg296_1.data_ptr()), c_void_p(arg297_1.data_ptr()))
    del arg296_1
    del arg297_1
    buf692 = reinterpret_tensor(buf689, (128, 512), (512, 1), 0); del buf689  # reuse
    # Source Nodes: [hidden_states_164], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg948_1, reinterpret_tensor(buf691, (128, 128), (128, 1), 0), reinterpret_tensor(arg947_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf692)
    del arg947_1
    del arg948_1
    buf693 = reinterpret_tensor(buf692, (1, 128, 512), (65536, 512, 1), 0); del buf692  # reuse
    cpp_fused_relu_223(c_void_p(buf693.data_ptr()))
    buf694 = buf690; del buf690  # reuse
    # Source Nodes: [layer_outputs_257], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg950_1, reinterpret_tensor(buf693, (128, 512), (512, 1), 0), reinterpret_tensor(arg949_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf694)
    del arg949_1
    del arg950_1
    buf695 = buf691; del buf691  # reuse
    cpp_fused_add_mul_224(c_void_p(buf695.data_ptr()), c_void_p(buf694.data_ptr()), c_void_p(arg298_1.data_ptr()), c_void_p(arg299_1.data_ptr()))
    del arg298_1
    del arg299_1
    buf696 = reinterpret_tensor(buf693, (128, 512), (512, 1), 0); del buf693  # reuse
    # Source Nodes: [hidden_states_166], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg952_1, reinterpret_tensor(buf695, (128, 128), (128, 1), 0), reinterpret_tensor(arg951_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf696)
    del arg951_1
    del arg952_1
    buf697 = reinterpret_tensor(buf696, (1, 128, 512), (65536, 512, 1), 0); del buf696  # reuse
    cpp_fused_relu_225(c_void_p(buf697.data_ptr()))
    buf698 = buf694; del buf694  # reuse
    # Source Nodes: [layer_outputs_260], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg954_1, reinterpret_tensor(buf697, (128, 512), (512, 1), 0), reinterpret_tensor(arg953_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf698)
    del arg953_1
    del arg954_1
    buf699 = buf695; del buf695  # reuse
    cpp_fused_add_mul_226(c_void_p(buf699.data_ptr()), c_void_p(buf698.data_ptr()), c_void_p(arg300_1.data_ptr()), c_void_p(arg301_1.data_ptr()))
    del arg300_1
    del arg301_1
    buf700 = reinterpret_tensor(buf697, (128, 512), (512, 1), 0); del buf697  # reuse
    # Source Nodes: [hidden_states_168], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg956_1, reinterpret_tensor(buf699, (128, 128), (128, 1), 0), reinterpret_tensor(arg955_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf700)
    del arg955_1
    del arg956_1
    buf701 = reinterpret_tensor(buf700, (1, 128, 512), (65536, 512, 1), 0); del buf700  # reuse
    cpp_fused_relu_227(c_void_p(buf701.data_ptr()))
    buf702 = buf698; del buf698  # reuse
    # Source Nodes: [layer_output_72], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg958_1, reinterpret_tensor(buf701, (128, 512), (512, 1), 0), reinterpret_tensor(arg957_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf702)
    del arg957_1
    del arg958_1
    buf703 = buf699; del buf699  # reuse
    cpp_fused_add_mul_228(c_void_p(buf703.data_ptr()), c_void_p(buf702.data_ptr()), c_void_p(arg302_1.data_ptr()), c_void_p(arg303_1.data_ptr()))
    del arg302_1
    del arg303_1
    buf704 = reinterpret_tensor(buf701, (128, 512), (512, 1), 0); del buf701  # reuse
    # Source Nodes: [layer_outputs_263], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg960_1, reinterpret_tensor(buf703, (128, 128), (128, 1), 0), reinterpret_tensor(arg959_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf704)
    del arg959_1
    del arg960_1
    buf705 = buf668; del buf668  # reuse
    cpp_fused_add_mul_229(c_void_p(buf705.data_ptr()), c_void_p(buf704.data_ptr()), c_void_p(arg304_1.data_ptr()), c_void_p(arg305_1.data_ptr()))
    del arg304_1
    del arg305_1
    buf706 = reinterpret_tensor(buf703, (128, 128), (128, 1), 0); del buf703  # reuse
    # Source Nodes: [layer_input_97], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg964_1, reinterpret_tensor(buf705, (128, 512), (512, 1), 0), reinterpret_tensor(arg963_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf706)
    del arg963_1
    del arg964_1
    buf707 = reinterpret_tensor(buf706, (1, 128, 128), (16384, 128, 1), 0); del buf706  # reuse
    cpp_fused_add_mul_230(c_void_p(buf707.data_ptr()), c_void_p(arg308_1.data_ptr()), c_void_p(arg309_1.data_ptr()))
    del arg308_1
    del arg309_1
    buf708 = buf702; del buf702  # reuse
    # Source Nodes: [mixed_query_layer_19], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg966_1, reinterpret_tensor(buf707, (128, 128), (128, 1), 0), reinterpret_tensor(arg965_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf708)
    del arg965_1
    del arg966_1
    buf709 = reinterpret_tensor(buf675, (128, 128), (128, 1), 0); del buf675  # reuse
    # Source Nodes: [mixed_key_layer_19], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg968_1, reinterpret_tensor(buf707, (128, 128), (128, 1), 0), reinterpret_tensor(arg967_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf709)
    del arg967_1
    del arg968_1
    buf710 = reinterpret_tensor(buf707, (128, 128), (128, 1), 0); del buf707  # reuse
    # Source Nodes: [mixed_value_layer_19], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg970_1, reinterpret_tensor(buf705, (128, 512), (512, 1), 0), reinterpret_tensor(arg969_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf710)
    del arg969_1
    del arg970_1
    buf711 = reinterpret_tensor(buf708, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf708  # reuse
    buf712 = reinterpret_tensor(buf709, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf709  # reuse
    buf713 = reinterpret_tensor(buf710, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf710  # reuse
    cpp_fused_231(c_void_p(buf711.data_ptr()), c_void_p(buf712.data_ptr()), c_void_p(buf713.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf714 = aten._scaled_dot_product_flash_attention(buf711, buf712, buf713, scale=0.17677669529663687)
    del buf711
    buf715 = buf714[0]
    del buf714
    buf722 = reinterpret_tensor(buf713, (128, 128), (128, 1), 0); del buf713  # reuse
    # Source Nodes: [layer_outputs_266], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg972_1, reinterpret_tensor(buf715, (128, 128), (128, 1), 0), reinterpret_tensor(arg971_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf722)
    del arg971_1
    del arg972_1
    buf723 = reinterpret_tensor(buf715, (128, 128), (128, 1), 0); del buf715  # reuse
    # Source Nodes: [layer_input_95], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg962_1, reinterpret_tensor(buf705, (128, 512), (512, 1), 0), reinterpret_tensor(arg961_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf723)
    del arg961_1
    del arg962_1
    buf724 = reinterpret_tensor(buf722, (1, 128, 128), (16384, 128, 1), 0); del buf722  # reuse
    cpp_fused_add_mul_232(c_void_p(buf724.data_ptr()), c_void_p(buf723.data_ptr()), c_void_p(arg306_1.data_ptr()), c_void_p(arg307_1.data_ptr()), c_void_p(arg310_1.data_ptr()), c_void_p(arg311_1.data_ptr()))
    del arg306_1
    del arg307_1
    del arg310_1
    del arg311_1
    buf725 = buf704; del buf704  # reuse
    # Source Nodes: [hidden_states_171], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg974_1, reinterpret_tensor(buf724, (128, 128), (128, 1), 0), reinterpret_tensor(arg973_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf725)
    del arg973_1
    del arg974_1
    buf726 = reinterpret_tensor(buf725, (1, 128, 512), (65536, 512, 1), 0); del buf725  # reuse
    cpp_fused_relu_233(c_void_p(buf726.data_ptr()))
    buf727 = buf723; del buf723  # reuse
    # Source Nodes: [layer_outputs_268], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg976_1, reinterpret_tensor(buf726, (128, 512), (512, 1), 0), reinterpret_tensor(arg975_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf727)
    del arg975_1
    del arg976_1
    buf728 = buf724; del buf724  # reuse
    cpp_fused_add_mul_234(c_void_p(buf728.data_ptr()), c_void_p(buf727.data_ptr()), c_void_p(arg312_1.data_ptr()), c_void_p(arg313_1.data_ptr()))
    del arg312_1
    del arg313_1
    buf729 = reinterpret_tensor(buf726, (128, 512), (512, 1), 0); del buf726  # reuse
    # Source Nodes: [hidden_states_173], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg978_1, reinterpret_tensor(buf728, (128, 128), (128, 1), 0), reinterpret_tensor(arg977_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf729)
    del arg977_1
    del arg978_1
    buf730 = reinterpret_tensor(buf729, (1, 128, 512), (65536, 512, 1), 0); del buf729  # reuse
    cpp_fused_relu_235(c_void_p(buf730.data_ptr()))
    buf731 = buf727; del buf727  # reuse
    # Source Nodes: [layer_outputs_271], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg980_1, reinterpret_tensor(buf730, (128, 512), (512, 1), 0), reinterpret_tensor(arg979_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf731)
    del arg979_1
    del arg980_1
    buf732 = buf728; del buf728  # reuse
    cpp_fused_add_mul_236(c_void_p(buf732.data_ptr()), c_void_p(buf731.data_ptr()), c_void_p(arg314_1.data_ptr()), c_void_p(arg315_1.data_ptr()))
    del arg314_1
    del arg315_1
    buf733 = reinterpret_tensor(buf730, (128, 512), (512, 1), 0); del buf730  # reuse
    # Source Nodes: [hidden_states_175], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg982_1, reinterpret_tensor(buf732, (128, 128), (128, 1), 0), reinterpret_tensor(arg981_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf733)
    del arg981_1
    del arg982_1
    buf734 = reinterpret_tensor(buf733, (1, 128, 512), (65536, 512, 1), 0); del buf733  # reuse
    cpp_fused_relu_237(c_void_p(buf734.data_ptr()))
    buf735 = buf731; del buf731  # reuse
    # Source Nodes: [layer_outputs_274], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg984_1, reinterpret_tensor(buf734, (128, 512), (512, 1), 0), reinterpret_tensor(arg983_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf735)
    del arg983_1
    del arg984_1
    buf736 = buf732; del buf732  # reuse
    cpp_fused_add_mul_238(c_void_p(buf736.data_ptr()), c_void_p(buf735.data_ptr()), c_void_p(arg316_1.data_ptr()), c_void_p(arg317_1.data_ptr()))
    del arg316_1
    del arg317_1
    buf737 = reinterpret_tensor(buf734, (128, 512), (512, 1), 0); del buf734  # reuse
    # Source Nodes: [hidden_states_177], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg986_1, reinterpret_tensor(buf736, (128, 128), (128, 1), 0), reinterpret_tensor(arg985_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf737)
    del arg985_1
    del arg986_1
    buf738 = reinterpret_tensor(buf737, (1, 128, 512), (65536, 512, 1), 0); del buf737  # reuse
    cpp_fused_relu_239(c_void_p(buf738.data_ptr()))
    buf739 = buf735; del buf735  # reuse
    # Source Nodes: [layer_output_76], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg988_1, reinterpret_tensor(buf738, (128, 512), (512, 1), 0), reinterpret_tensor(arg987_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf739)
    del arg987_1
    del arg988_1
    buf740 = buf736; del buf736  # reuse
    cpp_fused_add_mul_240(c_void_p(buf740.data_ptr()), c_void_p(buf739.data_ptr()), c_void_p(arg318_1.data_ptr()), c_void_p(arg319_1.data_ptr()))
    del arg318_1
    del arg319_1
    buf741 = reinterpret_tensor(buf738, (128, 512), (512, 1), 0); del buf738  # reuse
    # Source Nodes: [layer_outputs_277], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg990_1, reinterpret_tensor(buf740, (128, 128), (128, 1), 0), reinterpret_tensor(arg989_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf741)
    del arg989_1
    del arg990_1
    buf742 = buf705; del buf705  # reuse
    cpp_fused_add_mul_241(c_void_p(buf742.data_ptr()), c_void_p(buf741.data_ptr()), c_void_p(arg320_1.data_ptr()), c_void_p(arg321_1.data_ptr()))
    del arg320_1
    del arg321_1
    buf743 = reinterpret_tensor(buf740, (128, 128), (128, 1), 0); del buf740  # reuse
    # Source Nodes: [layer_input_102], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg994_1, reinterpret_tensor(buf742, (128, 512), (512, 1), 0), reinterpret_tensor(arg993_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf743)
    del arg993_1
    del arg994_1
    buf744 = reinterpret_tensor(buf743, (1, 128, 128), (16384, 128, 1), 0); del buf743  # reuse
    cpp_fused_add_mul_242(c_void_p(buf744.data_ptr()), c_void_p(arg324_1.data_ptr()), c_void_p(arg325_1.data_ptr()))
    del arg324_1
    del arg325_1
    buf745 = buf739; del buf739  # reuse
    # Source Nodes: [mixed_query_layer_20], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg996_1, reinterpret_tensor(buf744, (128, 128), (128, 1), 0), reinterpret_tensor(arg995_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf745)
    del arg995_1
    del arg996_1
    buf746 = reinterpret_tensor(buf712, (128, 128), (128, 1), 0); del buf712  # reuse
    # Source Nodes: [mixed_key_layer_20], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg998_1, reinterpret_tensor(buf744, (128, 128), (128, 1), 0), reinterpret_tensor(arg997_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf746)
    del arg997_1
    del arg998_1
    buf747 = reinterpret_tensor(buf744, (128, 128), (128, 1), 0); del buf744  # reuse
    # Source Nodes: [mixed_value_layer_20], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1000_1, reinterpret_tensor(buf742, (128, 512), (512, 1), 0), reinterpret_tensor(arg999_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf747)
    del arg1000_1
    del arg999_1
    buf748 = reinterpret_tensor(buf745, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf745  # reuse
    buf749 = reinterpret_tensor(buf746, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf746  # reuse
    buf750 = reinterpret_tensor(buf747, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf747  # reuse
    cpp_fused_243(c_void_p(buf748.data_ptr()), c_void_p(buf749.data_ptr()), c_void_p(buf750.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf751 = aten._scaled_dot_product_flash_attention(buf748, buf749, buf750, scale=0.17677669529663687)
    del buf748
    buf752 = buf751[0]
    del buf751
    buf759 = reinterpret_tensor(buf750, (128, 128), (128, 1), 0); del buf750  # reuse
    # Source Nodes: [layer_outputs_280], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1002_1, reinterpret_tensor(buf752, (128, 128), (128, 1), 0), reinterpret_tensor(arg1001_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf759)
    del arg1001_1
    del arg1002_1
    buf760 = reinterpret_tensor(buf752, (128, 128), (128, 1), 0); del buf752  # reuse
    # Source Nodes: [layer_input_100], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg992_1, reinterpret_tensor(buf742, (128, 512), (512, 1), 0), reinterpret_tensor(arg991_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf760)
    del arg991_1
    del arg992_1
    buf761 = reinterpret_tensor(buf759, (1, 128, 128), (16384, 128, 1), 0); del buf759  # reuse
    cpp_fused_add_mul_244(c_void_p(buf761.data_ptr()), c_void_p(buf760.data_ptr()), c_void_p(arg322_1.data_ptr()), c_void_p(arg323_1.data_ptr()), c_void_p(arg326_1.data_ptr()), c_void_p(arg327_1.data_ptr()))
    del arg322_1
    del arg323_1
    del arg326_1
    del arg327_1
    buf762 = buf741; del buf741  # reuse
    # Source Nodes: [hidden_states_180], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1004_1, reinterpret_tensor(buf761, (128, 128), (128, 1), 0), reinterpret_tensor(arg1003_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf762)
    del arg1003_1
    del arg1004_1
    buf763 = reinterpret_tensor(buf762, (1, 128, 512), (65536, 512, 1), 0); del buf762  # reuse
    cpp_fused_relu_245(c_void_p(buf763.data_ptr()))
    buf764 = buf760; del buf760  # reuse
    # Source Nodes: [layer_outputs_282], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1006_1, reinterpret_tensor(buf763, (128, 512), (512, 1), 0), reinterpret_tensor(arg1005_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf764)
    del arg1005_1
    del arg1006_1
    buf765 = buf761; del buf761  # reuse
    cpp_fused_add_mul_246(c_void_p(buf765.data_ptr()), c_void_p(buf764.data_ptr()), c_void_p(arg328_1.data_ptr()), c_void_p(arg329_1.data_ptr()))
    del arg328_1
    del arg329_1
    buf766 = reinterpret_tensor(buf763, (128, 512), (512, 1), 0); del buf763  # reuse
    # Source Nodes: [hidden_states_182], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1008_1, reinterpret_tensor(buf765, (128, 128), (128, 1), 0), reinterpret_tensor(arg1007_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf766)
    del arg1007_1
    del arg1008_1
    buf767 = reinterpret_tensor(buf766, (1, 128, 512), (65536, 512, 1), 0); del buf766  # reuse
    cpp_fused_relu_247(c_void_p(buf767.data_ptr()))
    buf768 = buf764; del buf764  # reuse
    # Source Nodes: [layer_outputs_285], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1010_1, reinterpret_tensor(buf767, (128, 512), (512, 1), 0), reinterpret_tensor(arg1009_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf768)
    del arg1009_1
    del arg1010_1
    buf769 = buf765; del buf765  # reuse
    cpp_fused_add_mul_248(c_void_p(buf769.data_ptr()), c_void_p(buf768.data_ptr()), c_void_p(arg330_1.data_ptr()), c_void_p(arg331_1.data_ptr()))
    del arg330_1
    del arg331_1
    buf770 = reinterpret_tensor(buf767, (128, 512), (512, 1), 0); del buf767  # reuse
    # Source Nodes: [hidden_states_184], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1012_1, reinterpret_tensor(buf769, (128, 128), (128, 1), 0), reinterpret_tensor(arg1011_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf770)
    del arg1011_1
    del arg1012_1
    buf771 = reinterpret_tensor(buf770, (1, 128, 512), (65536, 512, 1), 0); del buf770  # reuse
    cpp_fused_relu_249(c_void_p(buf771.data_ptr()))
    buf772 = buf768; del buf768  # reuse
    # Source Nodes: [layer_outputs_288], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1014_1, reinterpret_tensor(buf771, (128, 512), (512, 1), 0), reinterpret_tensor(arg1013_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf772)
    del arg1013_1
    del arg1014_1
    buf773 = buf769; del buf769  # reuse
    cpp_fused_add_mul_250(c_void_p(buf773.data_ptr()), c_void_p(buf772.data_ptr()), c_void_p(arg332_1.data_ptr()), c_void_p(arg333_1.data_ptr()))
    del arg332_1
    del arg333_1
    buf774 = reinterpret_tensor(buf771, (128, 512), (512, 1), 0); del buf771  # reuse
    # Source Nodes: [hidden_states_186], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1016_1, reinterpret_tensor(buf773, (128, 128), (128, 1), 0), reinterpret_tensor(arg1015_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf774)
    del arg1015_1
    del arg1016_1
    buf775 = reinterpret_tensor(buf774, (1, 128, 512), (65536, 512, 1), 0); del buf774  # reuse
    cpp_fused_relu_251(c_void_p(buf775.data_ptr()))
    buf776 = buf772; del buf772  # reuse
    # Source Nodes: [layer_output_80], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1018_1, reinterpret_tensor(buf775, (128, 512), (512, 1), 0), reinterpret_tensor(arg1017_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf776)
    del arg1017_1
    del arg1018_1
    buf777 = buf773; del buf773  # reuse
    cpp_fused_add_mul_252(c_void_p(buf777.data_ptr()), c_void_p(buf776.data_ptr()), c_void_p(arg334_1.data_ptr()), c_void_p(arg335_1.data_ptr()))
    del arg334_1
    del arg335_1
    buf778 = reinterpret_tensor(buf775, (128, 512), (512, 1), 0); del buf775  # reuse
    # Source Nodes: [layer_outputs_291], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1020_1, reinterpret_tensor(buf777, (128, 128), (128, 1), 0), reinterpret_tensor(arg1019_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf778)
    del arg1019_1
    del arg1020_1
    buf779 = buf742; del buf742  # reuse
    cpp_fused_add_mul_253(c_void_p(buf779.data_ptr()), c_void_p(buf778.data_ptr()), c_void_p(arg336_1.data_ptr()), c_void_p(arg337_1.data_ptr()))
    del arg336_1
    del arg337_1
    buf780 = reinterpret_tensor(buf777, (128, 128), (128, 1), 0); del buf777  # reuse
    # Source Nodes: [layer_input_107], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1024_1, reinterpret_tensor(buf779, (128, 512), (512, 1), 0), reinterpret_tensor(arg1023_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf780)
    del arg1023_1
    del arg1024_1
    buf781 = reinterpret_tensor(buf780, (1, 128, 128), (16384, 128, 1), 0); del buf780  # reuse
    cpp_fused_add_mul_254(c_void_p(buf781.data_ptr()), c_void_p(arg340_1.data_ptr()), c_void_p(arg341_1.data_ptr()))
    del arg340_1
    del arg341_1
    buf782 = buf776; del buf776  # reuse
    # Source Nodes: [mixed_query_layer_21], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1026_1, reinterpret_tensor(buf781, (128, 128), (128, 1), 0), reinterpret_tensor(arg1025_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf782)
    del arg1025_1
    del arg1026_1
    buf783 = reinterpret_tensor(buf749, (128, 128), (128, 1), 0); del buf749  # reuse
    # Source Nodes: [mixed_key_layer_21], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1028_1, reinterpret_tensor(buf781, (128, 128), (128, 1), 0), reinterpret_tensor(arg1027_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf783)
    del arg1027_1
    del arg1028_1
    buf784 = reinterpret_tensor(buf781, (128, 128), (128, 1), 0); del buf781  # reuse
    # Source Nodes: [mixed_value_layer_21], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1030_1, reinterpret_tensor(buf779, (128, 512), (512, 1), 0), reinterpret_tensor(arg1029_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf784)
    del arg1029_1
    del arg1030_1
    buf785 = reinterpret_tensor(buf782, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf782  # reuse
    buf786 = reinterpret_tensor(buf783, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf783  # reuse
    buf787 = reinterpret_tensor(buf784, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf784  # reuse
    cpp_fused_255(c_void_p(buf785.data_ptr()), c_void_p(buf786.data_ptr()), c_void_p(buf787.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf788 = aten._scaled_dot_product_flash_attention(buf785, buf786, buf787, scale=0.17677669529663687)
    del buf785
    buf789 = buf788[0]
    del buf788
    buf796 = reinterpret_tensor(buf787, (128, 128), (128, 1), 0); del buf787  # reuse
    # Source Nodes: [layer_outputs_294], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1032_1, reinterpret_tensor(buf789, (128, 128), (128, 1), 0), reinterpret_tensor(arg1031_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf796)
    del arg1031_1
    del arg1032_1
    buf797 = reinterpret_tensor(buf789, (128, 128), (128, 1), 0); del buf789  # reuse
    # Source Nodes: [layer_input_105], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1022_1, reinterpret_tensor(buf779, (128, 512), (512, 1), 0), reinterpret_tensor(arg1021_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf797)
    del arg1021_1
    del arg1022_1
    buf798 = reinterpret_tensor(buf796, (1, 128, 128), (16384, 128, 1), 0); del buf796  # reuse
    cpp_fused_add_mul_256(c_void_p(buf798.data_ptr()), c_void_p(buf797.data_ptr()), c_void_p(arg338_1.data_ptr()), c_void_p(arg339_1.data_ptr()), c_void_p(arg342_1.data_ptr()), c_void_p(arg343_1.data_ptr()))
    del arg338_1
    del arg339_1
    del arg342_1
    del arg343_1
    buf799 = buf778; del buf778  # reuse
    # Source Nodes: [hidden_states_189], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1034_1, reinterpret_tensor(buf798, (128, 128), (128, 1), 0), reinterpret_tensor(arg1033_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf799)
    del arg1033_1
    del arg1034_1
    buf800 = reinterpret_tensor(buf799, (1, 128, 512), (65536, 512, 1), 0); del buf799  # reuse
    cpp_fused_relu_257(c_void_p(buf800.data_ptr()))
    buf801 = buf797; del buf797  # reuse
    # Source Nodes: [layer_outputs_296], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1036_1, reinterpret_tensor(buf800, (128, 512), (512, 1), 0), reinterpret_tensor(arg1035_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf801)
    del arg1035_1
    del arg1036_1
    buf802 = buf798; del buf798  # reuse
    cpp_fused_add_mul_258(c_void_p(buf802.data_ptr()), c_void_p(buf801.data_ptr()), c_void_p(arg344_1.data_ptr()), c_void_p(arg345_1.data_ptr()))
    del arg344_1
    del arg345_1
    buf803 = reinterpret_tensor(buf800, (128, 512), (512, 1), 0); del buf800  # reuse
    # Source Nodes: [hidden_states_191], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1038_1, reinterpret_tensor(buf802, (128, 128), (128, 1), 0), reinterpret_tensor(arg1037_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf803)
    del arg1037_1
    del arg1038_1
    buf804 = reinterpret_tensor(buf803, (1, 128, 512), (65536, 512, 1), 0); del buf803  # reuse
    cpp_fused_relu_259(c_void_p(buf804.data_ptr()))
    buf805 = buf801; del buf801  # reuse
    # Source Nodes: [layer_outputs_299], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1040_1, reinterpret_tensor(buf804, (128, 512), (512, 1), 0), reinterpret_tensor(arg1039_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf805)
    del arg1039_1
    del arg1040_1
    buf806 = buf802; del buf802  # reuse
    cpp_fused_add_mul_260(c_void_p(buf806.data_ptr()), c_void_p(buf805.data_ptr()), c_void_p(arg346_1.data_ptr()), c_void_p(arg347_1.data_ptr()))
    del arg346_1
    del arg347_1
    buf807 = reinterpret_tensor(buf804, (128, 512), (512, 1), 0); del buf804  # reuse
    # Source Nodes: [hidden_states_193], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1042_1, reinterpret_tensor(buf806, (128, 128), (128, 1), 0), reinterpret_tensor(arg1041_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf807)
    del arg1041_1
    del arg1042_1
    buf808 = reinterpret_tensor(buf807, (1, 128, 512), (65536, 512, 1), 0); del buf807  # reuse
    cpp_fused_relu_261(c_void_p(buf808.data_ptr()))
    buf809 = buf805; del buf805  # reuse
    # Source Nodes: [layer_outputs_302], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1044_1, reinterpret_tensor(buf808, (128, 512), (512, 1), 0), reinterpret_tensor(arg1043_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf809)
    del arg1043_1
    del arg1044_1
    buf810 = buf806; del buf806  # reuse
    cpp_fused_add_mul_262(c_void_p(buf810.data_ptr()), c_void_p(buf809.data_ptr()), c_void_p(arg348_1.data_ptr()), c_void_p(arg349_1.data_ptr()))
    del arg348_1
    del arg349_1
    buf811 = reinterpret_tensor(buf808, (128, 512), (512, 1), 0); del buf808  # reuse
    # Source Nodes: [hidden_states_195], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1046_1, reinterpret_tensor(buf810, (128, 128), (128, 1), 0), reinterpret_tensor(arg1045_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf811)
    del arg1045_1
    del arg1046_1
    buf812 = reinterpret_tensor(buf811, (1, 128, 512), (65536, 512, 1), 0); del buf811  # reuse
    cpp_fused_relu_263(c_void_p(buf812.data_ptr()))
    buf813 = buf809; del buf809  # reuse
    # Source Nodes: [layer_output_84], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1048_1, reinterpret_tensor(buf812, (128, 512), (512, 1), 0), reinterpret_tensor(arg1047_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf813)
    del arg1047_1
    del arg1048_1
    buf814 = buf810; del buf810  # reuse
    cpp_fused_add_mul_264(c_void_p(buf814.data_ptr()), c_void_p(buf813.data_ptr()), c_void_p(arg350_1.data_ptr()), c_void_p(arg351_1.data_ptr()))
    del arg350_1
    del arg351_1
    buf815 = reinterpret_tensor(buf812, (128, 512), (512, 1), 0); del buf812  # reuse
    # Source Nodes: [layer_outputs_305], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1050_1, reinterpret_tensor(buf814, (128, 128), (128, 1), 0), reinterpret_tensor(arg1049_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf815)
    del arg1049_1
    del arg1050_1
    buf816 = buf779; del buf779  # reuse
    cpp_fused_add_mul_265(c_void_p(buf816.data_ptr()), c_void_p(buf815.data_ptr()), c_void_p(arg352_1.data_ptr()), c_void_p(arg353_1.data_ptr()))
    del arg352_1
    del arg353_1
    buf817 = reinterpret_tensor(buf814, (128, 128), (128, 1), 0); del buf814  # reuse
    # Source Nodes: [layer_input_112], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1054_1, reinterpret_tensor(buf816, (128, 512), (512, 1), 0), reinterpret_tensor(arg1053_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf817)
    del arg1053_1
    del arg1054_1
    buf818 = reinterpret_tensor(buf817, (1, 128, 128), (16384, 128, 1), 0); del buf817  # reuse
    cpp_fused_add_mul_266(c_void_p(buf818.data_ptr()), c_void_p(arg356_1.data_ptr()), c_void_p(arg357_1.data_ptr()))
    del arg356_1
    del arg357_1
    buf819 = buf813; del buf813  # reuse
    # Source Nodes: [mixed_query_layer_22], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1056_1, reinterpret_tensor(buf818, (128, 128), (128, 1), 0), reinterpret_tensor(arg1055_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf819)
    del arg1055_1
    del arg1056_1
    buf820 = reinterpret_tensor(buf786, (128, 128), (128, 1), 0); del buf786  # reuse
    # Source Nodes: [mixed_key_layer_22], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1058_1, reinterpret_tensor(buf818, (128, 128), (128, 1), 0), reinterpret_tensor(arg1057_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf820)
    del arg1057_1
    del arg1058_1
    buf821 = reinterpret_tensor(buf818, (128, 128), (128, 1), 0); del buf818  # reuse
    # Source Nodes: [mixed_value_layer_22], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1060_1, reinterpret_tensor(buf816, (128, 512), (512, 1), 0), reinterpret_tensor(arg1059_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf821)
    del arg1059_1
    del arg1060_1
    buf822 = reinterpret_tensor(buf819, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf819  # reuse
    buf823 = reinterpret_tensor(buf820, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf820  # reuse
    buf824 = reinterpret_tensor(buf821, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf821  # reuse
    cpp_fused_267(c_void_p(buf822.data_ptr()), c_void_p(buf823.data_ptr()), c_void_p(buf824.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf825 = aten._scaled_dot_product_flash_attention(buf822, buf823, buf824, scale=0.17677669529663687)
    del buf822
    buf826 = buf825[0]
    del buf825
    buf833 = reinterpret_tensor(buf824, (128, 128), (128, 1), 0); del buf824  # reuse
    # Source Nodes: [layer_outputs_308], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1062_1, reinterpret_tensor(buf826, (128, 128), (128, 1), 0), reinterpret_tensor(arg1061_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf833)
    del arg1061_1
    del arg1062_1
    buf834 = reinterpret_tensor(buf826, (128, 128), (128, 1), 0); del buf826  # reuse
    # Source Nodes: [layer_input_110], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1052_1, reinterpret_tensor(buf816, (128, 512), (512, 1), 0), reinterpret_tensor(arg1051_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf834)
    del arg1051_1
    del arg1052_1
    buf835 = reinterpret_tensor(buf833, (1, 128, 128), (16384, 128, 1), 0); del buf833  # reuse
    cpp_fused_add_mul_268(c_void_p(buf835.data_ptr()), c_void_p(buf834.data_ptr()), c_void_p(arg354_1.data_ptr()), c_void_p(arg355_1.data_ptr()), c_void_p(arg358_1.data_ptr()), c_void_p(arg359_1.data_ptr()))
    del arg354_1
    del arg355_1
    del arg358_1
    del arg359_1
    buf836 = buf815; del buf815  # reuse
    # Source Nodes: [hidden_states_198], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1064_1, reinterpret_tensor(buf835, (128, 128), (128, 1), 0), reinterpret_tensor(arg1063_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf836)
    del arg1063_1
    del arg1064_1
    buf837 = reinterpret_tensor(buf836, (1, 128, 512), (65536, 512, 1), 0); del buf836  # reuse
    cpp_fused_relu_269(c_void_p(buf837.data_ptr()))
    buf838 = buf834; del buf834  # reuse
    # Source Nodes: [layer_outputs_310], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1066_1, reinterpret_tensor(buf837, (128, 512), (512, 1), 0), reinterpret_tensor(arg1065_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf838)
    del arg1065_1
    del arg1066_1
    buf839 = buf835; del buf835  # reuse
    cpp_fused_add_mul_270(c_void_p(buf839.data_ptr()), c_void_p(buf838.data_ptr()), c_void_p(arg360_1.data_ptr()), c_void_p(arg361_1.data_ptr()))
    del arg360_1
    del arg361_1
    buf840 = reinterpret_tensor(buf837, (128, 512), (512, 1), 0); del buf837  # reuse
    # Source Nodes: [hidden_states_200], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1068_1, reinterpret_tensor(buf839, (128, 128), (128, 1), 0), reinterpret_tensor(arg1067_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf840)
    del arg1067_1
    del arg1068_1
    buf841 = reinterpret_tensor(buf840, (1, 128, 512), (65536, 512, 1), 0); del buf840  # reuse
    cpp_fused_relu_271(c_void_p(buf841.data_ptr()))
    buf842 = buf838; del buf838  # reuse
    # Source Nodes: [layer_outputs_313], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1070_1, reinterpret_tensor(buf841, (128, 512), (512, 1), 0), reinterpret_tensor(arg1069_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf842)
    del arg1069_1
    del arg1070_1
    buf843 = buf839; del buf839  # reuse
    cpp_fused_add_mul_272(c_void_p(buf843.data_ptr()), c_void_p(buf842.data_ptr()), c_void_p(arg362_1.data_ptr()), c_void_p(arg363_1.data_ptr()))
    del arg362_1
    del arg363_1
    buf844 = reinterpret_tensor(buf841, (128, 512), (512, 1), 0); del buf841  # reuse
    # Source Nodes: [hidden_states_202], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1072_1, reinterpret_tensor(buf843, (128, 128), (128, 1), 0), reinterpret_tensor(arg1071_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf844)
    del arg1071_1
    del arg1072_1
    buf845 = reinterpret_tensor(buf844, (1, 128, 512), (65536, 512, 1), 0); del buf844  # reuse
    cpp_fused_relu_273(c_void_p(buf845.data_ptr()))
    buf846 = buf842; del buf842  # reuse
    # Source Nodes: [layer_outputs_316], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1074_1, reinterpret_tensor(buf845, (128, 512), (512, 1), 0), reinterpret_tensor(arg1073_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf846)
    del arg1073_1
    del arg1074_1
    buf847 = buf843; del buf843  # reuse
    cpp_fused_add_mul_274(c_void_p(buf847.data_ptr()), c_void_p(buf846.data_ptr()), c_void_p(arg364_1.data_ptr()), c_void_p(arg365_1.data_ptr()))
    del arg364_1
    del arg365_1
    buf848 = reinterpret_tensor(buf845, (128, 512), (512, 1), 0); del buf845  # reuse
    # Source Nodes: [hidden_states_204], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1076_1, reinterpret_tensor(buf847, (128, 128), (128, 1), 0), reinterpret_tensor(arg1075_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf848)
    del arg1075_1
    del arg1076_1
    buf849 = reinterpret_tensor(buf848, (1, 128, 512), (65536, 512, 1), 0); del buf848  # reuse
    cpp_fused_relu_275(c_void_p(buf849.data_ptr()))
    buf850 = buf846; del buf846  # reuse
    # Source Nodes: [layer_output_88], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1078_1, reinterpret_tensor(buf849, (128, 512), (512, 1), 0), reinterpret_tensor(arg1077_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf850)
    del arg1077_1
    del arg1078_1
    buf851 = buf847; del buf847  # reuse
    cpp_fused_add_mul_276(c_void_p(buf851.data_ptr()), c_void_p(buf850.data_ptr()), c_void_p(arg366_1.data_ptr()), c_void_p(arg367_1.data_ptr()))
    del arg366_1
    del arg367_1
    buf852 = reinterpret_tensor(buf849, (128, 512), (512, 1), 0); del buf849  # reuse
    # Source Nodes: [layer_outputs_319], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1080_1, reinterpret_tensor(buf851, (128, 128), (128, 1), 0), reinterpret_tensor(arg1079_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf852)
    del arg1079_1
    del arg1080_1
    buf853 = buf816; del buf816  # reuse
    cpp_fused_add_mul_277(c_void_p(buf853.data_ptr()), c_void_p(buf852.data_ptr()), c_void_p(arg368_1.data_ptr()), c_void_p(arg369_1.data_ptr()))
    del arg368_1
    del arg369_1
    buf854 = reinterpret_tensor(buf851, (128, 128), (128, 1), 0); del buf851  # reuse
    # Source Nodes: [layer_input_117], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1084_1, reinterpret_tensor(buf853, (128, 512), (512, 1), 0), reinterpret_tensor(arg1083_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf854)
    del arg1083_1
    del arg1084_1
    buf855 = reinterpret_tensor(buf854, (1, 128, 128), (16384, 128, 1), 0); del buf854  # reuse
    cpp_fused_add_mul_278(c_void_p(buf855.data_ptr()), c_void_p(arg372_1.data_ptr()), c_void_p(arg373_1.data_ptr()))
    del arg372_1
    del arg373_1
    buf856 = buf850; del buf850  # reuse
    # Source Nodes: [mixed_query_layer_23], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1086_1, reinterpret_tensor(buf855, (128, 128), (128, 1), 0), reinterpret_tensor(arg1085_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf856)
    del arg1085_1
    del arg1086_1
    buf857 = reinterpret_tensor(buf823, (128, 128), (128, 1), 0); del buf823  # reuse
    # Source Nodes: [mixed_key_layer_23], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1088_1, reinterpret_tensor(buf855, (128, 128), (128, 1), 0), reinterpret_tensor(arg1087_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf857)
    del arg1087_1
    del arg1088_1
    buf858 = reinterpret_tensor(buf855, (128, 128), (128, 1), 0); del buf855  # reuse
    # Source Nodes: [mixed_value_layer_23], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1090_1, reinterpret_tensor(buf853, (128, 512), (512, 1), 0), reinterpret_tensor(arg1089_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf858)
    del arg1089_1
    del arg1090_1
    buf859 = reinterpret_tensor(buf856, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf856  # reuse
    buf860 = reinterpret_tensor(buf857, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf857  # reuse
    buf861 = reinterpret_tensor(buf858, (1, 4, 128, 32), (16384, 32, 128, 1), 0); del buf858  # reuse
    cpp_fused_279(c_void_p(buf859.data_ptr()), c_void_p(buf860.data_ptr()), c_void_p(buf861.data_ptr()))
    # Source Nodes: [], Original ATen: []
    buf862 = aten._scaled_dot_product_flash_attention(buf859, buf860, buf861, scale=0.17677669529663687)
    del buf859
    del buf860
    buf863 = buf862[0]
    del buf862
    buf870 = reinterpret_tensor(buf861, (128, 128), (128, 1), 0); del buf861  # reuse
    # Source Nodes: [layer_outputs_322], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1092_1, reinterpret_tensor(buf863, (128, 128), (128, 1), 0), reinterpret_tensor(arg1091_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf870)
    del arg1091_1
    del arg1092_1
    buf871 = reinterpret_tensor(buf863, (128, 128), (128, 1), 0); del buf863  # reuse
    # Source Nodes: [layer_input_115], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1082_1, reinterpret_tensor(buf853, (128, 512), (512, 1), 0), reinterpret_tensor(arg1081_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf871)
    del arg1081_1
    del arg1082_1
    buf872 = reinterpret_tensor(buf870, (1, 128, 128), (16384, 128, 1), 0); del buf870  # reuse
    cpp_fused_add_mul_280(c_void_p(buf872.data_ptr()), c_void_p(buf871.data_ptr()), c_void_p(arg370_1.data_ptr()), c_void_p(arg371_1.data_ptr()), c_void_p(arg374_1.data_ptr()), c_void_p(arg375_1.data_ptr()))
    del arg370_1
    del arg371_1
    del arg374_1
    del arg375_1
    buf873 = buf852; del buf852  # reuse
    # Source Nodes: [hidden_states_207], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1094_1, reinterpret_tensor(buf872, (128, 128), (128, 1), 0), reinterpret_tensor(arg1093_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf873)
    del arg1093_1
    del arg1094_1
    buf874 = reinterpret_tensor(buf873, (1, 128, 512), (65536, 512, 1), 0); del buf873  # reuse
    cpp_fused_relu_281(c_void_p(buf874.data_ptr()))
    buf875 = buf871; del buf871  # reuse
    # Source Nodes: [layer_outputs_324], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1096_1, reinterpret_tensor(buf874, (128, 512), (512, 1), 0), reinterpret_tensor(arg1095_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf875)
    del arg1095_1
    del arg1096_1
    buf876 = buf872; del buf872  # reuse
    cpp_fused_add_mul_282(c_void_p(buf876.data_ptr()), c_void_p(buf875.data_ptr()), c_void_p(arg376_1.data_ptr()), c_void_p(arg377_1.data_ptr()))
    del arg376_1
    del arg377_1
    buf877 = reinterpret_tensor(buf874, (128, 512), (512, 1), 0); del buf874  # reuse
    # Source Nodes: [hidden_states_209], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1098_1, reinterpret_tensor(buf876, (128, 128), (128, 1), 0), reinterpret_tensor(arg1097_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf877)
    del arg1097_1
    del arg1098_1
    buf878 = reinterpret_tensor(buf877, (1, 128, 512), (65536, 512, 1), 0); del buf877  # reuse
    cpp_fused_relu_283(c_void_p(buf878.data_ptr()))
    buf879 = buf875; del buf875  # reuse
    # Source Nodes: [layer_outputs_327], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1100_1, reinterpret_tensor(buf878, (128, 512), (512, 1), 0), reinterpret_tensor(arg1099_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf879)
    del arg1099_1
    del arg1100_1
    buf880 = buf876; del buf876  # reuse
    cpp_fused_add_mul_284(c_void_p(buf880.data_ptr()), c_void_p(buf879.data_ptr()), c_void_p(arg378_1.data_ptr()), c_void_p(arg379_1.data_ptr()))
    del arg378_1
    del arg379_1
    buf881 = reinterpret_tensor(buf878, (128, 512), (512, 1), 0); del buf878  # reuse
    # Source Nodes: [hidden_states_211], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1102_1, reinterpret_tensor(buf880, (128, 128), (128, 1), 0), reinterpret_tensor(arg1101_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf881)
    del arg1101_1
    del arg1102_1
    buf882 = reinterpret_tensor(buf881, (1, 128, 512), (65536, 512, 1), 0); del buf881  # reuse
    cpp_fused_relu_285(c_void_p(buf882.data_ptr()))
    buf883 = buf879; del buf879  # reuse
    # Source Nodes: [layer_outputs_330], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1104_1, reinterpret_tensor(buf882, (128, 512), (512, 1), 0), reinterpret_tensor(arg1103_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf883)
    del arg1103_1
    del arg1104_1
    buf884 = buf880; del buf880  # reuse
    cpp_fused_add_mul_286(c_void_p(buf884.data_ptr()), c_void_p(buf883.data_ptr()), c_void_p(arg380_1.data_ptr()), c_void_p(arg381_1.data_ptr()))
    del arg380_1
    del arg381_1
    buf885 = reinterpret_tensor(buf882, (128, 512), (512, 1), 0); del buf882  # reuse
    # Source Nodes: [hidden_states_213], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1106_1, reinterpret_tensor(buf884, (128, 128), (128, 1), 0), reinterpret_tensor(arg1105_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf885)
    del arg1105_1
    del arg1106_1
    buf886 = reinterpret_tensor(buf885, (1, 128, 512), (65536, 512, 1), 0); del buf885  # reuse
    cpp_fused_relu_287(c_void_p(buf886.data_ptr()))
    buf887 = buf883; del buf883  # reuse
    # Source Nodes: [layer_output_92], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1108_1, reinterpret_tensor(buf886, (128, 512), (512, 1), 0), reinterpret_tensor(arg1107_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf887)
    del arg1107_1
    del arg1108_1
    buf888 = buf884; del buf884  # reuse
    cpp_fused_add_mul_288(c_void_p(buf888.data_ptr()), c_void_p(buf887.data_ptr()), c_void_p(arg382_1.data_ptr()), c_void_p(arg383_1.data_ptr()))
    del arg382_1
    del arg383_1
    del buf887
    buf889 = reinterpret_tensor(buf886, (128, 512), (512, 1), 0); del buf886  # reuse
    # Source Nodes: [layer_outputs_333], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1110_1, reinterpret_tensor(buf888, (128, 128), (128, 1), 0), reinterpret_tensor(arg1109_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf889)
    del arg1109_1
    del arg1110_1
    del buf888
    buf890 = buf853; del buf853  # reuse
    cpp_fused_add_mul_289(c_void_p(buf890.data_ptr()), c_void_p(buf889.data_ptr()), c_void_p(arg384_1.data_ptr()), c_void_p(arg385_1.data_ptr()))
    del arg384_1
    del arg385_1
    del buf889
    buf891 = empty((128, 2), device='cpu', dtype=torch.float32)
    # Source Nodes: [logits], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg1112_1, reinterpret_tensor(buf890, (128, 512), (512, 1), 0), reinterpret_tensor(arg1111_1, (512, 2), (1, 512), 0), alpha=1, beta=1, out=buf891)
    del arg1111_1
    del arg1112_1
    del buf890
    buf892 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf893 = empty((1, 1), device='cpu', dtype=torch.float32)
    buf894 = empty((1, 1), device='cpu', dtype=torch.float32)
    buf895 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf896 = empty((1, 1), device='cpu', dtype=torch.float32)
    buf897 = empty((1, 1), device='cpu', dtype=torch.float32)
    buf898 = reinterpret_tensor(buf893, (), (), 0); del buf893  # reuse
    buf899 = buf898; del buf898  # reuse
    cpp_fused__log_softmax_add_clamp_clone_div_nll_loss_forward_290(c_void_p(buf899.data_ptr()), c_void_p(buf891.data_ptr()), c_void_p(arg1115_1.data_ptr()), c_void_p(arg1116_1.data_ptr()), c_void_p(buf892.data_ptr()), c_void_p(buf894.data_ptr()), c_void_p(buf895.data_ptr()), c_void_p(buf896.data_ptr()), c_void_p(buf897.data_ptr()))
    del arg1115_1
    del arg1116_1
    return (buf899, buf892, buf895, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg263_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg266_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg267_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg269_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg270_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg271_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg272_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg273_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg274_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg275_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg276_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg277_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg278_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg279_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg280_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg281_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg282_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg283_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg284_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg285_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg286_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg287_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg288_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg289_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg290_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg291_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg292_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg293_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg294_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg295_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg296_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg297_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg298_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg299_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg300_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg301_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg302_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg303_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg304_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg305_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg306_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg307_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg308_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg309_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg310_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg311_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg312_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg313_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg314_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg315_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg316_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg317_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg318_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg319_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg320_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg321_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg322_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg323_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg324_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg325_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg326_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg327_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg328_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg329_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg330_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg331_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg332_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg333_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg334_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg335_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg336_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg337_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg338_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg339_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg340_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg341_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg342_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg343_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg344_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg345_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg346_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg347_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg348_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg349_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg350_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg351_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg352_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg353_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg354_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg355_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg356_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg357_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg358_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg359_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg360_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg361_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg362_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg363_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg364_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg365_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg366_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg367_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg368_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg369_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg370_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg371_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg372_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg373_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg374_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg375_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg376_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg377_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg378_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg379_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg380_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg381_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg382_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg383_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg384_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg385_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg386_1 = rand_strided((30522, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg387_1 = rand_strided((512, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg388_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg389_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg390_1 = rand_strided((2, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg391_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg392_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg393_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg394_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg395_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg396_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg397_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg398_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg399_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg400_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg401_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg402_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg403_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg404_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg405_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg406_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg407_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg408_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg409_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg410_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg411_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg412_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg413_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg414_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg415_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg416_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg417_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg418_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg419_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg420_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg421_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg422_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg423_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg424_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg425_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg426_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg427_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg428_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg429_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg430_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg431_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg432_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg433_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg434_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg435_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg436_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg437_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg438_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg439_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg440_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg441_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg442_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg443_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg444_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg445_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg446_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg447_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg448_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg449_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg450_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg451_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg452_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg453_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg454_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg455_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg456_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg457_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg458_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg459_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg460_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg461_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg462_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg463_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg464_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg465_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg466_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg467_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg468_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg469_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg470_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg471_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg472_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg473_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg474_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg475_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg476_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg477_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg478_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg479_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg480_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg481_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg482_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg483_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg484_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg485_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg486_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg487_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg488_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg489_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg490_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg491_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg492_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg493_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg494_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg495_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg496_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg497_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg498_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg499_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg500_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg501_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg502_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg503_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg504_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg505_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg506_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg507_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg508_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg509_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg510_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg511_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg512_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg513_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg514_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg515_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg516_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg517_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg518_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg519_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg520_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg521_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg522_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg523_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg524_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg525_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg526_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg527_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg528_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg529_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg530_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg531_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg532_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg533_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg534_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg535_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg536_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg537_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg538_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg539_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg540_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg541_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg542_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg543_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg544_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg545_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg546_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg547_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg548_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg549_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg550_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg551_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg552_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg553_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg554_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg555_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg556_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg557_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg558_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg559_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg560_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg561_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg562_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg563_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg564_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg565_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg566_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg567_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg568_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg569_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg570_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg571_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg572_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg573_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg574_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg575_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg576_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg577_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg578_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg579_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg580_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg581_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg582_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg583_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg584_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg585_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg586_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg587_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg588_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg589_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg590_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg591_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg592_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg593_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg594_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg595_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg596_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg597_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg598_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg599_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg600_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg601_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg602_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg603_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg604_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg605_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg606_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg607_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg608_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg609_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg610_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg611_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg612_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg613_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg614_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg615_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg616_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg617_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg618_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg619_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg620_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg621_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg622_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg623_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg624_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg625_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg626_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg627_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg628_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg629_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg630_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg631_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg632_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg633_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg634_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg635_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg636_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg637_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg638_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg639_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg640_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg641_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg642_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg643_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg644_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg645_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg646_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg647_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg648_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg649_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg650_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg651_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg652_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg653_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg654_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg655_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg656_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg657_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg658_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg659_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg660_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg661_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg662_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg663_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg664_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg665_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg666_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg667_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg668_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg669_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg670_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg671_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg672_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg673_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg674_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg675_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg676_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg677_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg678_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg679_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg680_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg681_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg682_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg683_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg684_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg685_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg686_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg687_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg688_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg689_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg690_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg691_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg692_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg693_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg694_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg695_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg696_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg697_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg698_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg699_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg700_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg701_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg702_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg703_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg704_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg705_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg706_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg707_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg708_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg709_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg710_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg711_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg712_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg713_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg714_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg715_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg716_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg717_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg718_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg719_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg720_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg721_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg722_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg723_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg724_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg725_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg726_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg727_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg728_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg729_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg730_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg731_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg732_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg733_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg734_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg735_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg736_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg737_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg738_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg739_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg740_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg741_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg742_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg743_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg744_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg745_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg746_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg747_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg748_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg749_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg750_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg751_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg752_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg753_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg754_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg755_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg756_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg757_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg758_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg759_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg760_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg761_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg762_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg763_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg764_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg765_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg766_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg767_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg768_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg769_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg770_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg771_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg772_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg773_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg774_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg775_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg776_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg777_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg778_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg779_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg780_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg781_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg782_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg783_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg784_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg785_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg786_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg787_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg788_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg789_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg790_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg791_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg792_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg793_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg794_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg795_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg796_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg797_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg798_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg799_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg800_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg801_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg802_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg803_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg804_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg805_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg806_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg807_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg808_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg809_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg810_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg811_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg812_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg813_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg814_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg815_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg816_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg817_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg818_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg819_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg820_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg821_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg822_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg823_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg824_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg825_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg826_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg827_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg828_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg829_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg830_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg831_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg832_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg833_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg834_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg835_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg836_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg837_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg838_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg839_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg840_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg841_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg842_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg843_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg844_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg845_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg846_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg847_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg848_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg849_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg850_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg851_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg852_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg853_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg854_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg855_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg856_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg857_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg858_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg859_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg860_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg861_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg862_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg863_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg864_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg865_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg866_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg867_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg868_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg869_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg870_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg871_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg872_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg873_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg874_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg875_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg876_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg877_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg878_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg879_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg880_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg881_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg882_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg883_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg884_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg885_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg886_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg887_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg888_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg889_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg890_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg891_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg892_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg893_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg894_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg895_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg896_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg897_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg898_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg899_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg900_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg901_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg902_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg903_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg904_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg905_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg906_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg907_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg908_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg909_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg910_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg911_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg912_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg913_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg914_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg915_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg916_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg917_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg918_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg919_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg920_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg921_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg922_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg923_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg924_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg925_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg926_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg927_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg928_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg929_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg930_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg931_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg932_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg933_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg934_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg935_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg936_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg937_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg938_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg939_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg940_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg941_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg942_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg943_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg944_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg945_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg946_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg947_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg948_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg949_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg950_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg951_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg952_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg953_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg954_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg955_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg956_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg957_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg958_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg959_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg960_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg961_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg962_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg963_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg964_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg965_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg966_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg967_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg968_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg969_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg970_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg971_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg972_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg973_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg974_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg975_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg976_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg977_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg978_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg979_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg980_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg981_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg982_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg983_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg984_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg985_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg986_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg987_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg988_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg989_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg990_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg991_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg992_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg993_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg994_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg995_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg996_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg997_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg998_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg999_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg1000_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg1001_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg1002_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg1003_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg1004_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg1005_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg1006_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg1007_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg1008_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg1009_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg1010_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg1011_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg1012_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg1013_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg1014_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg1015_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg1016_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg1017_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg1018_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg1019_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg1020_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg1021_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg1022_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg1023_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg1024_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg1025_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg1026_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg1027_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg1028_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg1029_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg1030_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg1031_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg1032_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg1033_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg1034_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg1035_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg1036_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg1037_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg1038_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg1039_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg1040_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg1041_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg1042_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg1043_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg1044_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg1045_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg1046_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg1047_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg1048_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg1049_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg1050_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg1051_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg1052_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg1053_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg1054_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg1055_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg1056_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg1057_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg1058_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg1059_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg1060_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg1061_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg1062_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg1063_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg1064_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg1065_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg1066_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg1067_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg1068_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg1069_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg1070_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg1071_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg1072_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg1073_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg1074_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg1075_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg1076_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg1077_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg1078_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg1079_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg1080_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg1081_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg1082_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg1083_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg1084_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg1085_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg1086_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg1087_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg1088_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg1089_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg1090_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg1091_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg1092_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg1093_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg1094_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg1095_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg1096_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg1097_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg1098_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg1099_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg1100_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg1101_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg1102_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg1103_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg1104_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg1105_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg1106_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg1107_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg1108_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg1109_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg1110_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg1111_1 = rand_strided((2, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg1112_1 = rand_strided((2, ), (1, ), device='cpu', dtype=torch.float32)
    arg1113_1 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    arg1114_1 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    arg1115_1 = rand_strided((1, ), (1, ), device='cpu', dtype=torch.int64)
    arg1116_1 = rand_strided((1, ), (1, ), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1, arg797_1, arg798_1, arg799_1, arg800_1, arg801_1, arg802_1, arg803_1, arg804_1, arg805_1, arg806_1, arg807_1, arg808_1, arg809_1, arg810_1, arg811_1, arg812_1, arg813_1, arg814_1, arg815_1, arg816_1, arg817_1, arg818_1, arg819_1, arg820_1, arg821_1, arg822_1, arg823_1, arg824_1, arg825_1, arg826_1, arg827_1, arg828_1, arg829_1, arg830_1, arg831_1, arg832_1, arg833_1, arg834_1, arg835_1, arg836_1, arg837_1, arg838_1, arg839_1, arg840_1, arg841_1, arg842_1, arg843_1, arg844_1, arg845_1, arg846_1, arg847_1, arg848_1, arg849_1, arg850_1, arg851_1, arg852_1, arg853_1, arg854_1, arg855_1, arg856_1, arg857_1, arg858_1, arg859_1, arg860_1, arg861_1, arg862_1, arg863_1, arg864_1, arg865_1, arg866_1, arg867_1, arg868_1, arg869_1, arg870_1, arg871_1, arg872_1, arg873_1, arg874_1, arg875_1, arg876_1, arg877_1, arg878_1, arg879_1, arg880_1, arg881_1, arg882_1, arg883_1, arg884_1, arg885_1, arg886_1, arg887_1, arg888_1, arg889_1, arg890_1, arg891_1, arg892_1, arg893_1, arg894_1, arg895_1, arg896_1, arg897_1, arg898_1, arg899_1, arg900_1, arg901_1, arg902_1, arg903_1, arg904_1, arg905_1, arg906_1, arg907_1, arg908_1, arg909_1, arg910_1, arg911_1, arg912_1, arg913_1, arg914_1, arg915_1, arg916_1, arg917_1, arg918_1, arg919_1, arg920_1, arg921_1, arg922_1, arg923_1, arg924_1, arg925_1, arg926_1, arg927_1, arg928_1, arg929_1, arg930_1, arg931_1, arg932_1, arg933_1, arg934_1, arg935_1, arg936_1, arg937_1, arg938_1, arg939_1, arg940_1, arg941_1, arg942_1, arg943_1, arg944_1, arg945_1, arg946_1, arg947_1, arg948_1, arg949_1, arg950_1, arg951_1, arg952_1, arg953_1, arg954_1, arg955_1, arg956_1, arg957_1, arg958_1, arg959_1, arg960_1, arg961_1, arg962_1, arg963_1, arg964_1, arg965_1, arg966_1, arg967_1, arg968_1, arg969_1, arg970_1, arg971_1, arg972_1, arg973_1, arg974_1, arg975_1, arg976_1, arg977_1, arg978_1, arg979_1, arg980_1, arg981_1, arg982_1, arg983_1, arg984_1, arg985_1, arg986_1, arg987_1, arg988_1, arg989_1, arg990_1, arg991_1, arg992_1, arg993_1, arg994_1, arg995_1, arg996_1, arg997_1, arg998_1, arg999_1, arg1000_1, arg1001_1, arg1002_1, arg1003_1, arg1004_1, arg1005_1, arg1006_1, arg1007_1, arg1008_1, arg1009_1, arg1010_1, arg1011_1, arg1012_1, arg1013_1, arg1014_1, arg1015_1, arg1016_1, arg1017_1, arg1018_1, arg1019_1, arg1020_1, arg1021_1, arg1022_1, arg1023_1, arg1024_1, arg1025_1, arg1026_1, arg1027_1, arg1028_1, arg1029_1, arg1030_1, arg1031_1, arg1032_1, arg1033_1, arg1034_1, arg1035_1, arg1036_1, arg1037_1, arg1038_1, arg1039_1, arg1040_1, arg1041_1, arg1042_1, arg1043_1, arg1044_1, arg1045_1, arg1046_1, arg1047_1, arg1048_1, arg1049_1, arg1050_1, arg1051_1, arg1052_1, arg1053_1, arg1054_1, arg1055_1, arg1056_1, arg1057_1, arg1058_1, arg1059_1, arg1060_1, arg1061_1, arg1062_1, arg1063_1, arg1064_1, arg1065_1, arg1066_1, arg1067_1, arg1068_1, arg1069_1, arg1070_1, arg1071_1, arg1072_1, arg1073_1, arg1074_1, arg1075_1, arg1076_1, arg1077_1, arg1078_1, arg1079_1, arg1080_1, arg1081_1, arg1082_1, arg1083_1, arg1084_1, arg1085_1, arg1086_1, arg1087_1, arg1088_1, arg1089_1, arg1090_1, arg1091_1, arg1092_1, arg1093_1, arg1094_1, arg1095_1, arg1096_1, arg1097_1, arg1098_1, arg1099_1, arg1100_1, arg1101_1, arg1102_1, arg1103_1, arg1104_1, arg1105_1, arg1106_1, arg1107_1, arg1108_1, arg1109_1, arg1110_1, arg1111_1, arg1112_1, arg1113_1, arg1114_1, arg1115_1, arg1116_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('MobileBertForQuestionAnswering', benchmark_compiled_module)
