
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_clone_copy_embedding_eq_fill_lift_fresh_masked_fill_mul_native_layer_norm_new_zeros_select_scatter_slice_scatter_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp24_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp24_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp24_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp21 = tl.load(in_ptr2 + (2048 + r1 + (1024*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int32)
        tmp2 = tmp0 == tmp1
        tmp3 = tl.full([1, 1], 1, tl.int64)
        tmp4 = tmp0 >= tmp3
        tmp5 = tl.load(in_ptr0 + (tl.broadcast_to((-1) + x0, [XBLOCK, RBLOCK])), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp4, tmp5, tmp6)
        tmp8 = tl.full([1, 1], 0, tl.int64)
        tmp9 = tl.where(tmp4, tmp7, tmp8)
        tmp10 = tl.full([1, 1], 2, tl.int64)
        tmp11 = tl.where(tmp2, tmp10, tmp9)
        tmp12 = tl.full([1, 1], -100, tl.int64)
        tmp13 = tmp11 == tmp12
        tmp14 = tl.where(tmp13, tmp3, tmp11)
        tmp15 = tmp14 + 50265
        tmp16 = tmp14 < 0
        tmp17 = tl.where(tmp16, tmp15, tmp14)
        tl.device_assert((0 <= tmp17) & (tmp17 < 50265), "index out of bounds: 0 <= tmp17 < 50265")
        tmp18 = tl.load(in_ptr1 + (r1 + (1024*tmp17)), rmask, eviction_policy='evict_last', other=0.0)
        tmp19 = 1.0
        tmp20 = tmp18 * tmp19
        tmp22 = tmp20 + tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp24_mean_next, tmp24_m2_next, tmp24_weight_next = triton_helpers.welford_reduce(
            tmp23, tmp24_mean, tmp24_m2, tmp24_weight,
        )
        tmp24_mean = tl.where(rmask & xmask, tmp24_mean_next, tmp24_mean)
        tmp24_m2 = tl.where(rmask & xmask, tmp24_m2_next, tmp24_m2)
        tmp24_weight = tl.where(rmask & xmask, tmp24_weight_next, tmp24_weight)
    tmp24_tmp, tmp25_tmp, tmp26_tmp = triton_helpers.welford(
        tmp24_mean, tmp24_m2, tmp24_weight, 1
    )
    tmp24 = tmp24_tmp[:, None]
    tmp25 = tmp25_tmp[:, None]
    tmp26 = tmp26_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp48 = tl.load(in_ptr2 + (2048 + r1 + (1024*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp57 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp59 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp27 = x0
        tmp28 = tl.full([1, 1], 0, tl.int32)
        tmp29 = tmp27 == tmp28
        tmp30 = tl.full([1, 1], 1, tl.int64)
        tmp31 = tmp27 >= tmp30
        tmp32 = tl.load(in_ptr0 + (tl.broadcast_to((-1) + x0, [XBLOCK, RBLOCK])), rmask & tmp31 & xmask, eviction_policy='evict_last', other=0.0)
        tmp33 = tl.full(tmp32.shape, 0, tmp32.dtype)
        tmp34 = tl.where(tmp31, tmp32, tmp33)
        tmp35 = tl.full([1, 1], 0, tl.int64)
        tmp36 = tl.where(tmp31, tmp34, tmp35)
        tmp37 = tl.full([1, 1], 2, tl.int64)
        tmp38 = tl.where(tmp29, tmp37, tmp36)
        tmp39 = tl.full([1, 1], -100, tl.int64)
        tmp40 = tmp38 == tmp39
        tmp41 = tl.where(tmp40, tmp30, tmp38)
        tmp42 = tmp41 + 50265
        tmp43 = tmp41 < 0
        tmp44 = tl.where(tmp43, tmp42, tmp41)
        tl.device_assert((0 <= tmp44) & (tmp44 < 50265), "index out of bounds: 0 <= tmp44 < 50265")
        tmp45 = tl.load(in_ptr1 + (r1 + (1024*tmp44)), rmask, eviction_policy='evict_first', other=0.0)
        tmp46 = 1.0
        tmp47 = tmp45 * tmp46
        tmp49 = tmp47 + tmp48
        tmp50 = tmp49 - tmp24
        tmp51 = 1024.0
        tmp52 = tmp25 / tmp51
        tmp53 = 1e-05
        tmp54 = tmp52 + tmp53
        tmp55 = tl.math.rsqrt(tmp54)
        tmp56 = tmp50 * tmp55
        tmp58 = tmp56 * tmp57
        tmp60 = tmp58 + tmp59
        tl.store(out_ptr2 + (r1 + (1024*x0)), tmp60, rmask & xmask)
