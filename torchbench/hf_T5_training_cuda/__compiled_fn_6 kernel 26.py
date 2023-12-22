
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_embedding_dense_backward_sum_threshold_backward_25', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (x0), None)
    tmp3 = tl.load(in_ptr2 + (x0), None)
    tmp5 = tl.load(in_ptr3 + (x0), None)
    tmp7 = tl.load(in_ptr4 + (x0), None)
    tmp9 = tl.load(in_ptr5 + (x0), None)
    tmp11 = tl.load(in_ptr0 + (8388608 + x0), None)
    tmp12 = tl.load(in_ptr1 + (8388608 + x0), None)
    tmp14 = tl.load(in_ptr2 + (8388608 + x0), None)
    tmp16 = tl.load(in_ptr3 + (8388608 + x0), None)
    tmp18 = tl.load(in_ptr4 + (8388608 + x0), None)
    tmp20 = tl.load(in_ptr5 + (8388608 + x0), None)
    tmp23 = tl.load(in_ptr0 + (16777216 + x0), None)
    tmp24 = tl.load(in_ptr1 + (16777216 + x0), None)
    tmp26 = tl.load(in_ptr2 + (16777216 + x0), None)
    tmp28 = tl.load(in_ptr3 + (16777216 + x0), None)
    tmp30 = tl.load(in_ptr4 + (16777216 + x0), None)
    tmp32 = tl.load(in_ptr5 + (16777216 + x0), None)
    tmp35 = tl.load(in_ptr0 + (25165824 + x0), None)
    tmp36 = tl.load(in_ptr1 + (25165824 + x0), None)
    tmp38 = tl.load(in_ptr2 + (25165824 + x0), None)
    tmp40 = tl.load(in_ptr3 + (25165824 + x0), None)
    tmp42 = tl.load(in_ptr4 + (25165824 + x0), None)
    tmp44 = tl.load(in_ptr5 + (25165824 + x0), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp19 = tmp17 + tmp18
    tmp21 = tmp19 + tmp20
    tmp22 = tmp10 + tmp21
    tmp25 = tmp23 + tmp24
    tmp27 = tmp25 + tmp26
    tmp29 = tmp27 + tmp28
    tmp31 = tmp29 + tmp30
    tmp33 = tmp31 + tmp32
    tmp34 = tmp22 + tmp33
    tmp37 = tmp35 + tmp36
    tmp39 = tmp37 + tmp38
    tmp41 = tmp39 + tmp40
    tmp43 = tmp41 + tmp42
    tmp45 = tmp43 + tmp44
    tmp46 = tmp34 + tmp45
    tmp47 = tl.full([1], False, tl.int1)
    tmp48 = 0.0
    tmp49 = tl.where(tmp47, tmp48, tmp46)
    tl.store(in_out_ptr0 + (x0), tmp49, None)
