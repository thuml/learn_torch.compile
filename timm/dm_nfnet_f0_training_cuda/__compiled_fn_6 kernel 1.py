
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_gelu_gelu_backward_mul_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 884736
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 36)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x2), None)
    tmp1 = 36.0
    tmp2 = tmp0 / tmp1
    tmp3 = 1.7015043497085571
    tmp4 = tmp2 * tmp3
    tmp6 = 0.7071067811865476
    tmp7 = tmp5 * tmp6
    tmp8 = tl.math.erf(tmp7)
    tmp9 = 1.0
    tmp10 = tmp8 + tmp9
    tmp11 = 0.5
    tmp12 = tmp10 * tmp11
    tmp13 = tmp5 * tmp5
    tmp14 = -0.5
    tmp15 = tmp13 * tmp14
    tmp16 = tl.exp(tmp15)
    tmp17 = 0.3989422804014327
    tmp18 = tmp16 * tmp17
    tmp19 = tmp5 * tmp18
    tmp20 = tmp12 + tmp19
    tmp21 = tmp4 * tmp20
    tl.store(out_ptr0 + (x2), tmp21, None)
