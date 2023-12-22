
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 112
    x2 = (xindex // 7168)
    x3 = xindex
    x4 = (xindex // 7168) % 112
    tmp0 = tl.load(in_ptr0 + (x0 + (128*x1) + (28672*x2)), None)
    tmp1 = tl.load(in_ptr0 + (64 + x0 + (128*x1) + (28672*x2)), None)
    tmp3 = tl.load(in_ptr0 + (14336 + x0 + (128*x1) + (28672*x2)), None)
    tmp5 = tl.load(in_ptr0 + (14400 + x0 + (128*x1) + (28672*x2)), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = 1 + (2*x1) + (448*x4)
    tmp9 = (2*x1) + (448*x4)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = 224 + (2*x1) + (448*x4)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = 225 + (2*x1) + (448*x4)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x3), tmp6, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
