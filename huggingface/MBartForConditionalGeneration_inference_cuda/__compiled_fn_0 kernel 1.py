
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_mul_native_layer_norm_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp10_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp7 = tl.load(in_ptr2 + (2048 + r1 + (1024*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0 + 50265
        tmp2 = tmp0 < 0
        tmp3 = tl.where(tmp2, tmp1, tmp0)
        tl.device_assert(((0 <= tmp3) & (tmp3 < 50265)) | ~xmask, "index out of bounds: 0 <= tmp3 < 50265")
        tmp4 = tl.load(in_ptr1 + (r1 + (1024*tmp3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = 1.0
        tmp6 = tmp4 * tmp5
        tmp8 = tmp6 + tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp10_mean_next, tmp10_m2_next, tmp10_weight_next = triton_helpers.welford_reduce(
            tmp9, tmp10_mean, tmp10_m2, tmp10_weight,
        )
        tmp10_mean = tl.where(rmask & xmask, tmp10_mean_next, tmp10_mean)
        tmp10_m2 = tl.where(rmask & xmask, tmp10_m2_next, tmp10_m2)
        tmp10_weight = tl.where(rmask & xmask, tmp10_weight_next, tmp10_weight)
    tmp10_tmp, tmp11_tmp, tmp12_tmp = triton_helpers.welford(
        tmp10_mean, tmp10_m2, tmp10_weight, 1
    )
    tmp10 = tmp10_tmp[:, None]
    tmp11 = tmp11_tmp[:, None]
    tmp12 = tmp12_tmp[:, None]
    tmp33_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp33_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp33_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp19 = tl.load(in_ptr2 + (2048 + r1 + (1024*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp0 + 50265
        tmp14 = tmp0 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp0)
        tl.device_assert(((0 <= tmp15) & (tmp15 < 50265)) | ~xmask, "index out of bounds: 0 <= tmp15 < 50265")
        tmp16 = tl.load(in_ptr1 + (r1 + (1024*tmp15)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = 1.0
        tmp18 = tmp16 * tmp17
        tmp20 = tmp18 + tmp19
        tmp21 = tmp20 - tmp10
        tmp22 = 1024.0
        tmp23 = tmp11 / tmp22
        tmp24 = 1e-05
        tmp25 = tmp23 + tmp24
        tmp26 = tl.math.rsqrt(tmp25)
        tmp27 = tmp21 * tmp26
        tmp29 = tmp27 * tmp28
        tmp31 = tmp29 + tmp30
        tmp32 = tl.broadcast_to(tmp31, [XBLOCK, RBLOCK])
        tmp33_mean_next, tmp33_m2_next, tmp33_weight_next = triton_helpers.welford_reduce(
            tmp32, tmp33_mean, tmp33_m2, tmp33_weight,
        )
        tmp33_mean = tl.where(rmask & xmask, tmp33_mean_next, tmp33_mean)
        tmp33_m2 = tl.where(rmask & xmask, tmp33_m2_next, tmp33_m2)
        tmp33_weight = tl.where(rmask & xmask, tmp33_weight_next, tmp33_weight)
        tl.store(out_ptr2 + (r1 + (1024*x0)), tmp31, rmask & xmask)
    tmp33_tmp, tmp34_tmp, tmp35_tmp = triton_helpers.welford(
        tmp33_mean, tmp33_m2, tmp33_weight, 1
    )
    tmp33 = tmp33_tmp[:, None]
    tmp34 = tmp34_tmp[:, None]
    tmp35 = tmp35_tmp[:, None]
    tmp38_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp38_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp38_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp36 = tl.load(out_ptr2 + (r1 + (1024*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp37 = tl.broadcast_to(tmp36, [XBLOCK, RBLOCK])
        tmp38_mean_next, tmp38_m2_next, tmp38_weight_next = triton_helpers.welford_reduce(
            tmp37, tmp38_mean, tmp38_m2, tmp38_weight,
        )
        tmp38_mean = tl.where(rmask & xmask, tmp38_mean_next, tmp38_mean)
        tmp38_m2 = tl.where(rmask & xmask, tmp38_m2_next, tmp38_m2)
        tmp38_weight = tl.where(rmask & xmask, tmp38_weight_next, tmp38_weight)
    tmp38_tmp, tmp39_tmp, tmp40_tmp = triton_helpers.welford(
        tmp38_mean, tmp38_m2, tmp38_weight, 1
    )
    tmp38 = tmp38_tmp[:, None]
    tmp39 = tmp39_tmp[:, None]
    tmp40 = tmp40_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp41 = tl.load(out_ptr2 + (r1 + (1024*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp49 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp51 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp42 = tmp41 - tmp33
        tmp43 = 1024.0
        tmp44 = tmp39 / tmp43
        tmp45 = 1e-05
        tmp46 = tmp44 + tmp45
        tmp47 = tl.math.rsqrt(tmp46)
        tmp48 = tmp42 * tmp47
        tmp50 = tmp48 * tmp49
        tmp52 = tmp50 + tmp51
        tl.store(out_ptr5 + (r1 + (1024*x0)), tmp52, rmask & xmask)
