
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_gelu_native_layer_norm_21', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, rnumel):
    xnumel = 8
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (50432*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (50432*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tl.load(in_out_ptr0 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp40 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp50 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp52 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp81 = tl.load(in_ptr7 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp83 = tl.load(in_ptr8 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 256, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tl.full([1], 0, tl.int64)
    tmp22 = tmp21 >= tmp21
    tmp23 = tl.full([1], 1, tl.int64)
    tmp24 = tmp21 < tmp23
    tmp25 = tl.load(in_ptr3 + (r1 + (256*x0)), rmask & tmp24 & xmask, other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp24, tmp25, tmp26)
    tmp28 = tmp21 >= tmp23
    tmp29 = tl.full([1], 197, tl.int64)
    tmp30 = tmp21 < tmp29
    tmp31 = tl.load(in_ptr0 + (r1 + (50432*x0)), rmask & tmp28 & xmask, other=0.0)
    tmp32 = tl.load(in_ptr1 + (r1 + (50432*x0)), rmask & tmp28 & xmask, other=0.0)
    tmp33 = tl.load(in_ptr2 + (tl.broadcast_to(r1, [RBLOCK])), rmask & tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tmp32 + tmp33
    tmp35 = tmp31 + tmp34
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp28, tmp35, tmp36)
    tmp38 = tl.where(tmp24, tmp27, tmp37)
    tmp41 = tmp39 + tmp40
    tmp42 = tmp38 + tmp41
    tmp43 = tmp4 - tmp14
    tmp44 = 256.0
    tmp45 = tmp20 / tmp44
    tmp46 = 1e-06
    tmp47 = tmp45 + tmp46
    tmp48 = tl.math.rsqrt(tmp47)
    tmp49 = tmp43 * tmp48
    tmp51 = tmp49 * tmp50
    tmp53 = tmp51 + tmp52
    tmp54 = 0.5
    tmp55 = tmp53 * tmp54
    tmp56 = 0.7071067811865476
    tmp57 = tmp53 * tmp56
    tmp58 = tl.math.erf(tmp57)
    tmp59 = 1.0
    tmp60 = tmp58 + tmp59
    tmp61 = tmp55 * tmp60
    tmp62 = tl.broadcast_to(tmp42, [RBLOCK])
    tmp64 = tl.where(rmask & xmask, tmp62, 0)
    tmp65 = tl.broadcast_to(tmp62, [RBLOCK])
    tmp67 = tl.where(rmask & xmask, tmp65, 0)
    tmp68 = triton_helpers.promote_to_tensor(tl.sum(tmp67, 0))
    tmp69 = tmp68 / tmp13
    tmp70 = tmp62 - tmp69
    tmp71 = tmp70 * tmp70
    tmp72 = tl.broadcast_to(tmp71, [RBLOCK])
    tmp74 = tl.where(rmask & xmask, tmp72, 0)
    tmp75 = triton_helpers.promote_to_tensor(tl.sum(tmp74, 0))
    tmp76 = tmp42 - tmp69
    tmp77 = tmp75 / tmp44
    tmp78 = tmp77 + tmp46
    tmp79 = tl.math.rsqrt(tmp78)
    tmp80 = tmp76 * tmp79
    tmp82 = tmp80 * tmp81
    tmp84 = tmp82 + tmp83
    tmp85 = tmp84 * tmp54
    tmp86 = tmp84 * tmp56
    tmp87 = tl.math.erf(tmp86)
    tmp88 = tmp87 + tmp59
    tmp89 = tmp85 * tmp88
    tl.store(in_out_ptr0 + (r1 + (256*x0)), tmp42, rmask & xmask)
    tl.store(in_out_ptr1 + (r1 + (256*x0)), tmp61, rmask & xmask)
    tl.store(in_out_ptr2 + (r1 + (256*x0)), tmp89, rmask & xmask)
