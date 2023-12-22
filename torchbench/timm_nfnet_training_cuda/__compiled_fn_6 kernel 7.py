
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_gelu_gelu_backward_mul_sigmoid_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 221184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 36)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp6 = tl.load(in_ptr2 + (x2), None)
    tmp26 = tl.load(in_ptr3 + (0))
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK])
    tmp31 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = 0.9622504486493761
    tmp3 = tmp1 * tmp2
    tmp4 = 1.7015043497085571
    tmp5 = tmp3 * tmp4
    tmp7 = 0.7071067811865476
    tmp8 = tmp6 * tmp7
    tmp9 = tl.math.erf(tmp8)
    tmp10 = 1.0
    tmp11 = tmp9 + tmp10
    tmp12 = 0.5
    tmp13 = tmp11 * tmp12
    tmp14 = tmp6 * tmp6
    tmp15 = -0.5
    tmp16 = tmp14 * tmp15
    tmp17 = tl.exp(tmp16)
    tmp18 = 0.3989422804014327
    tmp19 = tmp17 * tmp18
    tmp20 = tmp6 * tmp19
    tmp21 = tmp13 + tmp20
    tmp22 = tmp5 * tmp21
    tmp23 = tmp0 + tmp22
    tmp24 = 0.2
    tmp25 = tmp23 * tmp24
    tmp28 = tmp25 * tmp27
    tmp29 = 2.0
    tmp30 = tmp28 * tmp29
    tmp32 = tl.sigmoid(tmp31)
    tmp33 = tmp30 * tmp32
    tmp35 = 36.0
    tmp36 = tmp34 / tmp35
    tmp37 = tmp33 + tmp36
    tl.store(out_ptr0 + (x2), tmp37, None)
