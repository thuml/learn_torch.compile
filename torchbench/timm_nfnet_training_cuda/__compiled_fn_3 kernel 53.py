
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_gelu_mul_52', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 519168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 13) % 13
    x0 = xindex % 13
    x2 = (xindex // 169)
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 12, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + (12*x1) + (144*x2)), tmp5 & xmask, other=0.0)
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = 0.7071067811865476
    tmp10 = tmp6 * tmp9
    tmp11 = tl.math.erf(tmp10)
    tmp12 = 1.0
    tmp13 = tmp11 + tmp12
    tmp14 = tmp8 * tmp13
    tmp15 = 1.7015043497085571
    tmp16 = tmp14 * tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp5, tmp16, tmp17)
    tl.store(out_ptr0 + (x3), tmp18, xmask)
