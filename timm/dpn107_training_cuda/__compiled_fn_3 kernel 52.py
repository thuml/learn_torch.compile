
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_51', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'in_ptr3', 'in_ptr4', 'out_ptr10', 'out_ptr4', 'out_ptr6', 'out_ptr8']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, out_ptr8, out_ptr10, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1152
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (903168*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tl.store(out_ptr1 + (x0), tmp3, xmask)
    tmp12 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = 6272.0
    tmp6 = tmp3 / tmp5
    tmp7 = 0.001
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp10 = 0.1
    tmp11 = tmp2 * tmp10
    tmp13 = 0.9
    tmp14 = tmp12 * tmp13
    tmp15 = tmp11 + tmp14
    tmp17 = tmp16 * tmp13
    tmp18 = tmp11 + tmp17
    tmp19 = 1.0001594642002871
    tmp20 = tmp6 * tmp19
    tmp21 = tmp20 * tmp10
    tmp23 = tmp22 * tmp13
    tmp24 = tmp21 + tmp23
    tmp26 = tmp25 * tmp13
    tmp27 = tmp21 + tmp26
    tl.store(out_ptr2 + (x0), tmp9, xmask)
    tl.store(out_ptr4 + (x0), tmp15, xmask)
    tl.store(out_ptr6 + (x0), tmp18, xmask)
    tl.store(out_ptr8 + (x0), tmp24, xmask)
    tl.store(out_ptr10 + (x0), tmp27, xmask)
