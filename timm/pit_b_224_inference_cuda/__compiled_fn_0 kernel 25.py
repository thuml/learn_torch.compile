
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_native_layer_norm_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4160
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 65
    x4 = (xindex // 65)
    x1 = (xindex // 65) % 8
    tmp20_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp20_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp20_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 1, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (r3 + (128*x4)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr1 + (r3 + (128*x1)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
        tmp9 = tl.where(tmp4, tmp7, tmp8)
        tmp10 = tmp0 >= tmp3
        tmp11 = tl.full([1, 1], 65, tl.int64)
        tmp12 = tmp0 < tmp11
        tmp13 = tl.load(in_ptr2 + ((64*r3) + (8192*x4) + (((-1) + x0) % 64)), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr3 + (r3 + (128*x1)), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp13 + tmp14
        tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
        tmp17 = tl.where(tmp10, tmp15, tmp16)
        tmp18 = tl.where(tmp4, tmp9, tmp17)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp20_mean_next, tmp20_m2_next, tmp20_weight_next = triton_helpers.welford_reduce(
            tmp19, tmp20_mean, tmp20_m2, tmp20_weight,
        )
        tmp20_mean = tl.where(rmask & xmask, tmp20_mean_next, tmp20_mean)
        tmp20_m2 = tl.where(rmask & xmask, tmp20_m2_next, tmp20_m2)
        tmp20_weight = tl.where(rmask & xmask, tmp20_weight_next, tmp20_weight)
    tmp20_tmp, tmp21_tmp, tmp22_tmp = triton_helpers.welford(
        tmp20_mean, tmp20_m2, tmp20_weight, 1
    )
    tmp20 = tmp20_tmp[:, None]
    tmp21 = tmp21_tmp[:, None]
    tmp22 = tmp22_tmp[:, None]
    tl.store(out_ptr0 + (x5), tmp20, xmask)
    tl.store(out_ptr1 + (x5), tmp21, xmask)
    tl.store(out_ptr2 + (x5), tmp22, xmask)
