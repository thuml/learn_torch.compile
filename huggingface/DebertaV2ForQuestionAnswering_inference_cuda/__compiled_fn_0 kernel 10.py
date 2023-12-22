
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*i64', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_nll_loss_forward_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp12 = tl.load(in_out_ptr0 + (0))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK])
    tmp15 = tl.load(in_ptr2 + (0))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp25 = tl.load(in_ptr3 + (0))
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK])
    tmp35 = tl.load(in_ptr5 + (0))
    tmp36 = tl.broadcast_to(tmp35, [XBLOCK])
    tmp38 = tl.load(in_ptr6 + (0))
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK])
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp4 = tl.full([1], 512, tl.int64)
    tmp5 = triton_helpers.minimum(tmp3, tmp4)
    tmp6 = tmp5 != tmp4
    tmp7 = tl.where(tmp6, tmp5, tmp2)
    tmp8 = tmp7 + 512
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert((0 <= tmp10) & (tmp10 < 512), "index out of bounds: 0 <= tmp10 < 512")
    tmp11 = tl.load(in_ptr1 + (tmp10), None, eviction_policy='evict_last')
    tmp14 = tmp11 - tmp13
    tmp17 = tl.log(tmp16)
    tmp18 = tmp14 - tmp17
    tmp19 = -tmp18
    tmp20 = 0.0
    tmp21 = tl.where(tmp6, tmp19, tmp20)
    tmp22 = tmp6.to(tl.int64)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 / tmp23
    tmp27 = triton_helpers.maximum(tmp26, tmp2)
    tmp28 = triton_helpers.minimum(tmp27, tmp4)
    tmp29 = tmp28 != tmp4
    tmp30 = tl.where(tmp29, tmp28, tmp2)
    tmp31 = tmp30 + 512
    tmp32 = tmp30 < 0
    tmp33 = tl.where(tmp32, tmp31, tmp30)
    tl.device_assert((0 <= tmp33) & (tmp33 < 512), "index out of bounds: 0 <= tmp33 < 512")
    tmp34 = tl.load(in_ptr4 + (tmp33), None, eviction_policy='evict_last')
    tmp37 = tmp34 - tmp36
    tmp40 = tl.log(tmp39)
    tmp41 = tmp37 - tmp40
    tmp42 = -tmp41
    tmp43 = tl.where(tmp29, tmp42, tmp20)
    tmp44 = tmp29.to(tl.int64)
    tmp45 = tmp44.to(tl.float32)
    tmp46 = tmp43 / tmp45
    tmp47 = tmp24 + tmp46
    tmp48 = 2.0
    tmp49 = tmp47 / tmp48
    tl.store(in_out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp49, None)
