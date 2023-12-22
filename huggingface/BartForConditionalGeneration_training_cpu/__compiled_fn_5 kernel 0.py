
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


cpp_fused_clone_copy_eq_fill_lift_fresh_masked_fill_new_zeros_select_scatter_slice_scatter_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const long* in_ptr0,
                       long* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = c10::convert<int>(x0);
            auto tmp1 = static_cast<int>(0);
            auto tmp2 = tmp0 == tmp1;
            auto tmp3 = c10::convert<long>(x0);
            auto tmp4 = static_cast<long>(1);
            auto tmp5 = tmp3 >= tmp4;
            auto tmp6 = [&]
            {
                auto tmp7 = in_ptr0[static_cast<long>((-1L) + x0)];
                return tmp7;
            }
            ;
            auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0);
            auto tmp9 = static_cast<long>(0);
            auto tmp10 = tmp5 ? tmp8 : tmp9;
            auto tmp11 = static_cast<long>(2);
            auto tmp12 = tmp2 ? tmp11 : tmp10;
            auto tmp13 = static_cast<long>(-100);
            auto tmp14 = tmp12 == tmp13;
            auto tmp15 = [&]
            {
                auto tmp16 = in_ptr0[static_cast<long>((-1L) + x0)];
                return tmp16;
            }
            ;
            auto tmp17 = tmp5 ? tmp15() : static_cast<decltype(tmp15())>(0);
            auto tmp18 = tmp5 ? tmp17 : tmp9;
            auto tmp19 = tmp2 ? tmp11 : tmp18;
            auto tmp20 = tmp14 ? tmp4 : tmp19;
            out_ptr0[static_cast<long>(x0)] = tmp20;
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (1, 1024), (1024, 1))
    buf0 = empty((1, 1024), device='cpu', dtype=torch.int64)
    cpp_fused_clone_copy_eq_fill_lift_fresh_masked_fill_new_zeros_select_scatter_slice_scatter_0(c_void_p(arg0_1.data_ptr()), c_void_p(buf0.data_ptr()))
    del arg0_1
    return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 1024), (1024, 1), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('BartForConditionalGeneration', benchmark_compiled_module)
