
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_46', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 40
    rnumel = 7527
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 5
    x1 = (xindex // 5)
    tmp21_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp21_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp21_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7527*x0)
        tmp1 = tl.full([1, 1], 37632, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((37632*x1) + ((r2 + (7527*x0)) % 37632)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + ((37632*x1) + ((r2 + (7527*x0)) % 37632)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (((r2 + (7527*x0)) // 49) % 768), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = tl.load(in_ptr3 + (((r2 + (7527*x0)) // 49) % 768), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 * tmp7
        tmp9 = tmp3 + tmp8
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = 0.0
        tmp13 = tl.full(tmp12.shape, 0, tmp12.dtype)
        tmp14 = tl.where(tmp2, tmp12, tmp13)
        tmp15 = 1.0
        tmp16 = tl.full(tmp15.shape, 0, tmp15.dtype)
        tmp17 = tl.where(tmp2, tmp15, tmp16)
        tmp18 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp19 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp20 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp21_mean_next, tmp21_m2_next, tmp21_weight_next = triton_helpers.welford_combine(
            tmp21_mean, tmp21_m2, tmp21_weight,
            tmp18, tmp19, tmp20
        )
        tmp21_mean = tl.where(rmask & xmask, tmp21_mean_next, tmp21_mean)
        tmp21_m2 = tl.where(rmask & xmask, tmp21_m2_next, tmp21_m2)
        tmp21_weight = tl.where(rmask & xmask, tmp21_weight_next, tmp21_weight)
    tmp21_tmp, tmp22_tmp, tmp23_tmp = triton_helpers.welford(
        tmp21_mean, tmp21_m2, tmp21_weight, 1
    )
    tmp21 = tmp21_tmp[:, None]
    tmp22 = tmp22_tmp[:, None]
    tmp23 = tmp23_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp21, xmask)
    tl.store(out_ptr1 + (x3), tmp22, xmask)
    tl.store(out_ptr2 + (x3), tmp23, xmask)
