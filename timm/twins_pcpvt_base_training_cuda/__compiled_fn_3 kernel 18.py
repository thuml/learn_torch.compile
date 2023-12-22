
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_view_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x1 = (xindex // 784)
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (784*r2) + (100352*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight,
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tmp22_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp22_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp22_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp7 = tl.load(in_ptr0 + (x0 + (784*r2) + (100352*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp7 + tmp8
        tmp10 = tmp9 - tmp4
        tmp11 = 128.0
        tmp12 = tmp5 / tmp11
        tmp13 = 1e-05
        tmp14 = tmp12 + tmp13
        tmp15 = tl.math.rsqrt(tmp14)
        tmp16 = tmp10 * tmp15
        tmp18 = tmp16 * tmp17
        tmp20 = tmp18 + tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp22_mean_next, tmp22_m2_next, tmp22_weight_next = triton_helpers.welford_reduce(
            tmp21, tmp22_mean, tmp22_m2, tmp22_weight,
        )
        tmp22_mean = tl.where(rmask & xmask, tmp22_mean_next, tmp22_mean)
        tmp22_m2 = tl.where(rmask & xmask, tmp22_m2_next, tmp22_m2)
        tmp22_weight = tl.where(rmask & xmask, tmp22_weight_next, tmp22_weight)
        tl.store(out_ptr2 + (r2 + (128*x3)), tmp16, rmask & xmask)
    tmp22_tmp, tmp23_tmp, tmp24_tmp = triton_helpers.welford(
        tmp22_mean, tmp22_m2, tmp22_weight, 1
    )
    tmp22 = tmp22_tmp[:, None]
    tmp23 = tmp23_tmp[:, None]
    tmp24 = tmp24_tmp[:, None]
    tmp31_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp31_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp31_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp25 = tl.load(out_ptr2 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp26 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp28 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp27 = tmp25 * tmp26
        tmp29 = tmp27 + tmp28
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
        tmp31_mean_next, tmp31_m2_next, tmp31_weight_next = triton_helpers.welford_reduce(
            tmp30, tmp31_mean, tmp31_m2, tmp31_weight,
        )
        tmp31_mean = tl.where(rmask & xmask, tmp31_mean_next, tmp31_mean)
        tmp31_m2 = tl.where(rmask & xmask, tmp31_m2_next, tmp31_m2)
        tmp31_weight = tl.where(rmask & xmask, tmp31_weight_next, tmp31_weight)
    tmp31_tmp, tmp32_tmp, tmp33_tmp = triton_helpers.welford(
        tmp31_mean, tmp31_m2, tmp31_weight, 1
    )
    tmp31 = tmp31_tmp[:, None]
    tmp32 = tmp32_tmp[:, None]
    tmp33 = tmp33_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp34 = tl.load(out_ptr2 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp35 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp37 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp46 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp48 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp36 = tmp34 * tmp35
        tmp38 = tmp36 + tmp37
        tmp39 = tmp38 - tmp22
        tmp40 = 128.0
        tmp41 = tmp32 / tmp40
        tmp42 = 1e-06
        tmp43 = tmp41 + tmp42
        tmp44 = tl.math.rsqrt(tmp43)
        tmp45 = tmp39 * tmp44
        tmp47 = tmp45 * tmp46
        tmp49 = tmp47 + tmp48
        tl.store(out_ptr5 + (r2 + (128*x3)), tmp45, rmask & xmask)
        tl.store(out_ptr6 + (r2 + (128*x3)), tmp49, rmask & xmask)
        tl.store(out_ptr7 + (r2 + (128*x3)), tmp49, rmask & xmask)
    tmp50 = 128.0
    tmp51 = tmp32 / tmp50
    tmp52 = 1e-06
    tmp53 = tmp51 + tmp52
    tmp54 = tl.math.rsqrt(tmp53)
    tmp55 = tmp54 / tmp50
    tmp56 = tmp5 / tmp50
    tmp57 = 1e-05
    tmp58 = tmp56 + tmp57
    tmp59 = tl.math.rsqrt(tmp58)
    tmp60 = tmp59 / tmp50
    tl.store(out_ptr8 + (x3), tmp55, xmask)
    tl.store(out_ptr9 + (x3), tmp60, xmask)
