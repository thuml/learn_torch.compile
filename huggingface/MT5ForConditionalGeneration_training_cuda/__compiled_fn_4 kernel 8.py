
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_native_dropout_backward_pow_tanh_backward_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (x0), None).to(tl.int1)
    tmp6 = tl.load(in_ptr2 + (x0), None)
    tmp9 = tl.load(in_ptr3 + (x0), None)
    tmp14 = tl.load(in_ptr4 + (x0), None)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp10 = 1.0
    tmp11 = tmp9 + tmp10
    tmp12 = tmp8 * tmp11
    tmp13 = tmp5 * tmp12
    tmp15 = tmp5 * tmp14
    tmp16 = tmp15 * tmp8
    tmp17 = tmp9 * tmp9
    tmp18 = tmp10 - tmp17
    tmp19 = tmp16 * tmp18
    tmp20 = 0.7978845608028654
    tmp21 = tmp19 * tmp20
    tmp22 = 0.044715
    tmp23 = tmp21 * tmp22
    tmp24 = tmp6 * tmp6
    tmp25 = 3.0
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tmp28 = tmp21 + tmp27
    tmp29 = tmp15 * tmp11
    tmp30 = tmp29 * tmp7
    tmp31 = tmp28 + tmp30
    tl.store(out_ptr0 + (x0), tmp13, None)
    tl.store(in_out_ptr0 + (x0), tmp31, None)
