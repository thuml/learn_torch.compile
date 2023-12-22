
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_mul_native_layer_norm_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp23_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp23_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp23_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp1 = tmp0 + 128112
        tmp2 = tmp0 < 0
        tmp3 = tl.where(tmp2, tmp1, tmp0)
        tl.device_assert(((0 <= tmp3) & (tmp3 < 128112)) | ~xmask, "index out of bounds: 0 <= tmp3 < 128112")
        tmp4 = tl.load(in_ptr1 + (r1 + (1024*tmp3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = 32.0
        tmp6 = tmp4 * tmp5
        tmp8 = tmp7.to(tl.int32)
        tmp9 = tl.full([1, 1], 0, tl.int32)
        tmp10 = tmp8 + tmp9
        tmp11 = tl.full([1, 1], 1, tl.int64)
        tmp12 = tmp0 != tmp11
        tmp13 = tmp12.to(tl.int32)
        tmp14 = tmp10 * tmp13
        tmp15 = tmp14.to(tl.int64)
        tmp16 = tmp15 + tmp11
        tmp17 = tmp16 + 1026
        tmp18 = tmp16 < 0
        tmp19 = tl.where(tmp18, tmp17, tmp16)
        tl.device_assert(((0 <= tmp19) & (tmp19 < 1026)) | ~xmask, "index out of bounds: 0 <= tmp19 < 1026")
        tmp20 = tl.load(in_ptr3 + (r1 + (1024*tmp19)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tmp6 + tmp20
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp23_mean_next, tmp23_m2_next, tmp23_weight_next = triton_helpers.welford_reduce(
            tmp22, tmp23_mean, tmp23_m2, tmp23_weight,
        )
        tmp23_mean = tl.where(rmask & xmask, tmp23_mean_next, tmp23_mean)
        tmp23_m2 = tl.where(rmask & xmask, tmp23_m2_next, tmp23_m2)
        tmp23_weight = tl.where(rmask & xmask, tmp23_weight_next, tmp23_weight)
    tmp23_tmp, tmp24_tmp, tmp25_tmp = triton_helpers.welford(
        tmp23_mean, tmp23_m2, tmp23_weight, 1
    )
    tmp23 = tmp23_tmp[:, None]
    tmp24 = tmp24_tmp[:, None]
    tmp25 = tmp25_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp53 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp55 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp26 = tmp0 + 128112
        tmp27 = tmp0 < 0
        tmp28 = tl.where(tmp27, tmp26, tmp0)
        tl.device_assert(((0 <= tmp28) & (tmp28 < 128112)) | ~xmask, "index out of bounds: 0 <= tmp28 < 128112")
        tmp29 = tl.load(in_ptr1 + (r1 + (1024*tmp28)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp30 = 32.0
        tmp31 = tmp29 * tmp30
        tmp32 = tmp7.to(tl.int32)
        tmp33 = tl.full([1, 1], 0, tl.int32)
        tmp34 = tmp32 + tmp33
        tmp35 = tl.full([1, 1], 1, tl.int64)
        tmp36 = tmp0 != tmp35
        tmp37 = tmp36.to(tl.int32)
        tmp38 = tmp34 * tmp37
        tmp39 = tmp38.to(tl.int64)
        tmp40 = tmp39 + tmp35
        tmp41 = tmp40 + 1026
        tmp42 = tmp40 < 0
        tmp43 = tl.where(tmp42, tmp41, tmp40)
        tl.device_assert(((0 <= tmp43) & (tmp43 < 1026)) | ~xmask, "index out of bounds: 0 <= tmp43 < 1026")
        tmp44 = tl.load(in_ptr3 + (r1 + (1024*tmp43)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp45 = tmp31 + tmp44
        tmp46 = tmp45 - tmp23
        tmp47 = 1024.0
        tmp48 = tmp24 / tmp47
        tmp49 = 1e-05
        tmp50 = tmp48 + tmp49
        tmp51 = tl.math.rsqrt(tmp50)
        tmp52 = tmp46 * tmp51
        tmp54 = tmp52 * tmp53
        tmp56 = tmp54 + tmp55
        tl.store(out_ptr2 + (r1 + (1024*x0)), tmp56, rmask & xmask)
