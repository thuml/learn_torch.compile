
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_native_layer_norm_34', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6280
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 785
    x1 = (xindex // 785)
    x3 = xindex
    tmp24_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp24_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp24_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp19 = tl.load(in_ptr3 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 1, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + ((785*r2) + (100480*x1)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
        tmp7 = tl.where(tmp4, tmp5, tmp6)
        tmp8 = tmp0 >= tmp3
        tmp9 = tl.full([1, 1], 785, tl.int64)
        tmp10 = tmp0 < tmp9
        tmp11 = tl.load(in_ptr1 + ((784*r2) + (100352*x1) + (((-1) + x0) % 784)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp11 + tmp12
        tmp14 = tl.load(in_ptr0 + (1 + (785*r2) + (100480*x1) + (((-1) + x0) % 784)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp13 + tmp14
        tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
        tmp17 = tl.where(tmp8, tmp15, tmp16)
        tmp18 = tl.where(tmp4, tmp7, tmp17)
        tmp21 = tmp19 + tmp20
        tmp22 = tmp18 + tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp24_mean_next, tmp24_m2_next, tmp24_weight_next = triton_helpers.welford_reduce(
            tmp23, tmp24_mean, tmp24_m2, tmp24_weight,
        )
        tmp24_mean = tl.where(rmask & xmask, tmp24_mean_next, tmp24_mean)
        tmp24_m2 = tl.where(rmask & xmask, tmp24_m2_next, tmp24_m2)
        tmp24_weight = tl.where(rmask & xmask, tmp24_weight_next, tmp24_weight)
        tl.store(out_ptr0 + (x0 + (785*r2) + (100480*x1)), tmp22, rmask & xmask)
    tmp24_tmp, tmp25_tmp, tmp26_tmp = triton_helpers.welford(
        tmp24_mean, tmp24_m2, tmp24_weight, 1
    )
    tmp24 = tmp24_tmp[:, None]
    tmp25 = tmp25_tmp[:, None]
    tmp26 = tmp26_tmp[:, None]
    tmp29_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp29_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp29_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp27 = tl.load(out_ptr0 + (x0 + (785*r2) + (100480*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
        tmp29_mean_next, tmp29_m2_next, tmp29_weight_next = triton_helpers.welford_reduce(
            tmp28, tmp29_mean, tmp29_m2, tmp29_weight,
        )
        tmp29_mean = tl.where(rmask & xmask, tmp29_mean_next, tmp29_mean)
        tmp29_m2 = tl.where(rmask & xmask, tmp29_m2_next, tmp29_m2)
        tmp29_weight = tl.where(rmask & xmask, tmp29_weight_next, tmp29_weight)
    tmp29_tmp, tmp30_tmp, tmp31_tmp = triton_helpers.welford(
        tmp29_mean, tmp29_m2, tmp29_weight, 1
    )
    tmp29 = tmp29_tmp[:, None]
    tmp30 = tmp30_tmp[:, None]
    tmp31 = tmp31_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp32 = tl.load(out_ptr0 + (x0 + (785*r2) + (100480*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp40 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp42 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp33 = tmp32 - tmp24
        tmp34 = 128.0
        tmp35 = tmp30 / tmp34
        tmp36 = 1e-06
        tmp37 = tmp35 + tmp36
        tmp38 = tl.math.rsqrt(tmp37)
        tmp39 = tmp33 * tmp38
        tmp41 = tmp39 * tmp40
        tmp43 = tmp41 + tmp42
        tl.store(out_ptr3 + (r2 + (128*x3)), tmp43, rmask & xmask)
