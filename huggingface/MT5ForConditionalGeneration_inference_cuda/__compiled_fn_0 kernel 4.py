
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_mean_mul_pow_rsqrt_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp5 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0 + 250112
        tmp2 = tmp0 < 0
        tmp3 = tl.where(tmp2, tmp1, tmp0)
        tl.device_assert(((0 <= tmp3) & (tmp3 < 250112)) | ~xmask, "index out of bounds: 0 <= tmp3 < 250112")
        tmp4 = tl.load(in_ptr1 + (r1 + (512*tmp3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = tmp6 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
        tmp12 = tmp11 + 250112
        tmp13 = tmp11 < 0
        tmp14 = tl.where(tmp13, tmp12, tmp11)
        tl.device_assert(((0 <= tmp14) & (tmp14 < 250112)) | ~xmask, "index out of bounds: 0 <= tmp14 < 250112")
        tmp15 = tl.load(in_ptr1 + (r1 + (512*tmp14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tmp15 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp20 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp25 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tmp0 + 250112
        tmp22 = tmp0 < 0
        tmp23 = tl.where(tmp22, tmp21, tmp0)
        tl.device_assert(((0 <= tmp23) & (tmp23 < 250112)) | ~xmask, "index out of bounds: 0 <= tmp23 < 250112")
        tmp24 = tl.load(in_ptr1 + (r1 + (512*tmp23)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp26 = tmp24 + tmp25
        tmp27 = 512.0
        tmp28 = tmp9 / tmp27
        tmp29 = 1e-06
        tmp30 = tmp28 + tmp29
        tmp31 = tl.math.rsqrt(tmp30)
        tmp32 = tmp26 * tmp31
        tmp33 = tmp20 * tmp32
        tmp35 = tmp11 + 250112
        tmp36 = tmp11 < 0
        tmp37 = tl.where(tmp36, tmp35, tmp11)
        tl.device_assert(((0 <= tmp37) & (tmp37 < 250112)) | ~xmask, "index out of bounds: 0 <= tmp37 < 250112")
        tmp38 = tl.load(in_ptr1 + (r1 + (512*tmp37)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp39 = tmp18 / tmp27
        tmp40 = tmp39 + tmp29
        tmp41 = tl.math.rsqrt(tmp40)
        tmp42 = tmp38 * tmp41
        tmp43 = tmp34 * tmp42
        tl.store(out_ptr2 + (r1 + (512*x0)), tmp33, rmask & xmask)
        tl.store(out_ptr3 + (r1 + (512*x0)), tmp43, rmask & xmask)
