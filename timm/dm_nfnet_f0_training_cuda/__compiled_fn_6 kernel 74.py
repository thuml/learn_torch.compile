
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_gelu_gelu_backward_mul_73', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp7 = tl.load(in_ptr1 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tmp5 = 1.7015043497085571
    tmp6 = tmp4 * tmp5
    tmp8 = 0.7071067811865476
    tmp9 = tmp7 * tmp8
    tmp10 = tl.math.erf(tmp9)
    tmp11 = tmp10 + tmp3
    tmp12 = 0.5
    tmp13 = tmp11 * tmp12
    tmp14 = tmp7 * tmp7
    tmp15 = -0.5
    tmp16 = tmp14 * tmp15
    tmp17 = tl.exp(tmp16)
    tmp18 = 0.3989422804014327
    tmp19 = tmp17 * tmp18
    tmp20 = tmp7 * tmp19
    tmp21 = tmp13 + tmp20
    tmp22 = tmp6 * tmp21
    tl.store(in_out_ptr0 + (x0), tmp22, None)
