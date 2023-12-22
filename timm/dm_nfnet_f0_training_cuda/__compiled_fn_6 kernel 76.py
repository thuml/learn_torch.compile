
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_gelu_gelu_backward_mul_75', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4718592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 96) % 96
    x0 = xindex % 96
    x2 = (xindex // 9216)
    x3 = xindex
    tmp11 = tl.load(in_ptr1 + (x3), None)
    tmp0 = x1
    tmp1 = tl.full([1], 97, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + (97*x1) + (9409*x2)), tmp5, other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = 1.7015043497085571
    tmp10 = tmp8 * tmp9
    tmp12 = 0.7071067811865476
    tmp13 = tmp11 * tmp12
    tmp14 = tl.math.erf(tmp13)
    tmp15 = 1.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.5
    tmp18 = tmp16 * tmp17
    tmp19 = tmp11 * tmp11
    tmp20 = -0.5
    tmp21 = tmp19 * tmp20
    tmp22 = tl.exp(tmp21)
    tmp23 = 0.3989422804014327
    tmp24 = tmp22 * tmp23
    tmp25 = tmp11 * tmp24
    tmp26 = tmp18 + tmp25
    tmp27 = tmp10 * tmp26
    tl.store(out_ptr0 + (x3), tmp27, None)
