
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13, 14))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_mul_native_layer_norm_native_layer_norm_backward_44', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel):
    xnumel = 6280
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 785
    r2 = rindex
    x1 = (xindex // 785)
    x3 = xindex
    tmp15 = tl.load(in_ptr2 + (r2 + (768*x3)), rmask & xmask, other=0.0)
    tmp26 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp52 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp54 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (768*x1)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 785, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + (r2 + (768*x3)), rmask & tmp8 & xmask, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp16 = tl.load(in_ptr3 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.load(in_ptr4 + (r2 + (768*x1)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp4, tmp18, tmp19)
    tmp21 = tl.load(in_ptr2 + (r2 + (768*x3)), rmask & tmp8 & xmask, other=0.0)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp8, tmp21, tmp22)
    tmp24 = tl.where(tmp4, tmp20, tmp23)
    tmp25 = tmp15 + tmp24
    tmp27 = tmp26 * tmp14
    tmp28 = tmp25 + tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = tl.broadcast_to(tmp29, [RBLOCK])
    tmp34 = tl.where(rmask & xmask, tmp32, 0)
    tmp35 = triton_helpers.promote_to_tensor(tl.sum(tmp34, 0))
    tmp36 = tl.full([1], 768, tl.int32)
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp35 / tmp37
    tmp39 = tmp29 - tmp38
    tmp40 = tmp39 * tmp39
    tmp41 = tl.broadcast_to(tmp40, [RBLOCK])
    tmp43 = tl.where(rmask & xmask, tmp41, 0)
    tmp44 = triton_helpers.promote_to_tensor(tl.sum(tmp43, 0))
    tmp45 = tmp28 - tmp38
    tmp46 = 768.0
    tmp47 = tmp44 / tmp46
    tmp48 = 1e-06
    tmp49 = tmp47 + tmp48
    tmp50 = tl.math.rsqrt(tmp49)
    tmp51 = tmp45 * tmp50
    tmp53 = tmp51 * tmp52
    tmp55 = tmp53 + tmp54
    tmp56 = tmp50 / tmp46
    tl.store(out_ptr0 + (r2 + (768*x3)), tmp14, rmask & xmask)
    tl.store(out_ptr1 + (r2 + (768*x3)), tmp28, rmask & xmask)
    tl.store(out_ptr4 + (r2 + (768*x3)), tmp51, rmask & xmask)
    tl.store(out_ptr5 + (r2 + (768*x3)), tmp55, rmask & xmask)
    tl.store(out_ptr6 + (x3), tmp56, xmask)
