
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*i1', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_div_embedding_masked_fill_mean_mul_pow_rsub_sqrt_sub_1', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp1 = tmp0 + 50265
        tmp2 = tmp0 < 0
        tmp3 = tl.where(tmp2, tmp1, tmp0)
        tl.device_assert(((0 <= tmp3) & (tmp3 < 50265)) | ~xmask, "index out of bounds: 0 <= tmp3 < 50265")
        tmp4 = tl.load(in_ptr1 + (r1 + (768*tmp3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp5 + 512
        tmp7 = tmp5 < 0
        tmp8 = tl.where(tmp7, tmp6, tmp5)
        tl.device_assert(((0 <= tmp8) & (tmp8 < 512)) | ~xmask, "index out of bounds: 0 <= tmp8 < 512")
        tmp9 = tl.load(in_ptr3 + (r1 + (768*tmp8)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp4 + tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    _tmp28 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp14 = tmp0 + 50265
        tmp15 = tmp0 < 0
        tmp16 = tl.where(tmp15, tmp14, tmp0)
        tl.device_assert(((0 <= tmp16) & (tmp16 < 50265)) | ~xmask, "index out of bounds: 0 <= tmp16 < 50265")
        tmp17 = tl.load(in_ptr1 + (r1 + (768*tmp16)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tmp5 + 512
        tmp19 = tmp5 < 0
        tmp20 = tl.where(tmp19, tmp18, tmp5)
        tl.device_assert(((0 <= tmp20) & (tmp20 < 512)) | ~xmask, "index out of bounds: 0 <= tmp20 < 512")
        tmp21 = tl.load(in_ptr3 + (r1 + (768*tmp20)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp22 = tmp17 + tmp21
        tmp23 = 768.0
        tmp24 = tmp12 / tmp23
        tmp25 = tmp22 - tmp24
        tmp26 = tmp25 * tmp25
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp29 = _tmp28 + tmp27
        _tmp28 = tl.where(rmask & xmask, tmp29, _tmp28)
        tl.store(out_ptr1 + (r1 + (768*x0)), tmp25, rmask & xmask)
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    tmp30 = 768.0
    tmp31 = tmp28 / tmp30
    tmp32 = 1e-07
    tmp33 = tmp31 + tmp32
    tmp34 = tl.sqrt(tmp33)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp34, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp35 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp39 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp40 = tl.load(out_ptr1 + (r1 + (768*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp43 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp36 = 1.0
        tmp37 = tmp36 - tmp35
        tmp38 = (tmp37 != 0)
        tmp41 = tmp40 / tmp34
        tmp42 = tmp39 * tmp41
        tmp44 = tmp42 + tmp43
        tmp45 = 0.0
        tmp46 = tl.where(tmp38, tmp45, tmp44)
        tmp47 = 1.1111111111111112
        tmp48 = tmp46 * tmp47
        tl.store(out_ptr2 + (r1 + (768*x0)), tmp38, rmask & xmask)
        tl.store(out_ptr3 + (r1 + (768*x0)), tmp48, rmask & xmask)
