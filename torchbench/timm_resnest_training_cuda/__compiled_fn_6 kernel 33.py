
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[64], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_32', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp5 = tl.load(in_ptr0 + (64 + x0), xmask)
    tmp7 = tl.load(in_ptr1 + (64 + x0), xmask)
    tmp10 = tl.load(in_ptr0 + (128 + x0), xmask)
    tmp12 = tl.load(in_ptr1 + (128 + x0), xmask)
    tmp15 = tl.load(in_ptr0 + (192 + x0), xmask)
    tmp17 = tl.load(in_ptr1 + (192 + x0), xmask)
    tmp20 = tl.load(in_ptr2 + (x0), xmask)
    tmp21 = tl.load(in_ptr3 + (x0), xmask)
    tmp24 = tl.load(in_ptr2 + (64 + x0), xmask)
    tmp28 = tl.load(in_ptr2 + (128 + x0), xmask)
    tmp32 = tl.load(in_ptr2 + (192 + x0), xmask)
    tmp36 = tl.load(in_ptr4 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = tmp5 <= tmp1
    tmp8 = tl.where(tmp6, tmp1, tmp7)
    tmp9 = tmp4 + tmp8
    tmp11 = tmp10 <= tmp1
    tmp13 = tl.where(tmp11, tmp1, tmp12)
    tmp14 = tmp9 + tmp13
    tmp16 = tmp15 <= tmp1
    tmp18 = tl.where(tmp16, tmp1, tmp17)
    tmp19 = tmp14 + tmp18
    tmp22 = tmp20 - tmp21
    tmp23 = tmp4 * tmp22
    tmp25 = tmp24 - tmp21
    tmp26 = tmp8 * tmp25
    tmp27 = tmp23 + tmp26
    tmp29 = tmp28 - tmp21
    tmp30 = tmp13 * tmp29
    tmp31 = tmp27 + tmp30
    tmp33 = tmp32 - tmp21
    tmp34 = tmp18 * tmp33
    tmp35 = tmp31 + tmp34
    tmp37 = 1e-05
    tmp38 = tmp36 + tmp37
    tmp39 = tl.math.rsqrt(tmp38)
    tmp40 = tmp35 * tmp39
    tl.store(out_ptr0 + (x0), tmp19, xmask)
    tl.store(in_out_ptr0 + (x0), tmp40, xmask)
