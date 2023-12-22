
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_native_layer_norm_46', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4728
    rnumel = 107
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 197) % 3
    x0 = xindex % 197
    x2 = (xindex // 591)
    tmp35_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp35_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp35_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (107*x1)
        tmp1 = tl.full([1, 1], 320, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.broadcast_to(x0, [XBLOCK, RBLOCK])
        tmp4 = tl.full([1, 1], 0, tl.int64)
        tmp5 = tmp3 >= tmp4
        tmp6 = tl.full([1, 1], 1, tl.int64)
        tmp7 = tmp3 < tmp6
        tmp8 = tmp7 & tmp2
        tmp9 = tl.load(in_ptr0 + ((197*r3) + (21079*x1) + (63040*x2)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
        tmp11 = tl.where(tmp8, tmp9, tmp10)
        tmp12 = tmp3 >= tmp6
        tmp13 = tl.full([1, 1], 197, tl.int64)
        tmp14 = tmp3 < tmp13
        tmp15 = tmp12 & tmp2
        tmp16 = tl.load(in_ptr1 + ((196*r3) + (20972*x1) + (62720*x2) + (((-1) + x0) % 196)), rmask & tmp15 & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr2 + (r3 + (107*x1)), rmask & tmp15 & xmask, eviction_policy='evict_last', other=0.0)
        tmp18 = tmp16 + tmp17
        tmp19 = tl.load(in_ptr0 + (1 + (197*r3) + (21079*x1) + (63040*x2) + (((-1) + x0) % 196)), rmask & tmp15 & xmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tmp18 + tmp19
        tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
        tmp22 = tl.where(tmp15, tmp20, tmp21)
        tmp23 = tl.where(tmp7, tmp11, tmp22)
        tmp24 = tl.full(tmp23.shape, 0, tmp23.dtype)
        tmp25 = tl.where(tmp2, tmp23, tmp24)
        tmp26 = 0.0
        tmp27 = tl.full(tmp26.shape, 0, tmp26.dtype)
        tmp28 = tl.where(tmp2, tmp26, tmp27)
        tmp29 = 1.0
        tmp30 = tl.full(tmp29.shape, 0, tmp29.dtype)
        tmp31 = tl.where(tmp2, tmp29, tmp30)
        tmp32 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
        tmp33 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
        tmp34 = tl.broadcast_to(tmp31, [XBLOCK, RBLOCK])
        tmp35_mean_next, tmp35_m2_next, tmp35_weight_next = triton_helpers.welford_combine(
            tmp35_mean, tmp35_m2, tmp35_weight,
            tmp32, tmp33, tmp34
        )
        tmp35_mean = tl.where(rmask & xmask, tmp35_mean_next, tmp35_mean)
        tmp35_m2 = tl.where(rmask & xmask, tmp35_m2_next, tmp35_m2)
        tmp35_weight = tl.where(rmask & xmask, tmp35_weight_next, tmp35_weight)
    tmp35_tmp, tmp36_tmp, tmp37_tmp = triton_helpers.welford(
        tmp35_mean, tmp35_m2, tmp35_weight, 1
    )
    tmp35 = tmp35_tmp[:, None]
    tmp36 = tmp36_tmp[:, None]
    tmp37 = tmp37_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp35, xmask)
    tl.store(out_ptr1 + (x4), tmp36, xmask)
    tl.store(out_ptr2 + (x4), tmp37, xmask)
