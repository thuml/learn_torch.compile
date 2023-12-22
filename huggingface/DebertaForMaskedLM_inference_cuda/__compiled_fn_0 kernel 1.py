
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_embedding_mean_mul_pow_sqrt_sub_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
        tmp17 = tl.load(in_ptr1 + (r1 + (768*tmp16)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp18 = tmp5 + 512
        tmp19 = tmp5 < 0
        tmp20 = tl.where(tmp19, tmp18, tmp5)
        tl.device_assert(((0 <= tmp20) & (tmp20 < 512)) | ~xmask, "index out of bounds: 0 <= tmp20 < 512")
        tmp21 = tl.load(in_ptr3 + (r1 + (768*tmp20)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp22 = tmp17 + tmp21
        tmp23 = 768.0
        tmp24 = tmp12 / tmp23
        tmp25 = tmp22 - tmp24
        tmp26 = tmp25 * tmp25
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp29 = _tmp28 + tmp27
        _tmp28 = tl.where(rmask & xmask, tmp29, _tmp28)
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp49 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp31 = tmp0 + 50265
        tmp32 = tmp0 < 0
        tmp33 = tl.where(tmp32, tmp31, tmp0)
        tl.device_assert(((0 <= tmp33) & (tmp33 < 50265)) | ~xmask, "index out of bounds: 0 <= tmp33 < 50265")
        tmp34 = tl.load(in_ptr1 + (r1 + (768*tmp33)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp35 = tmp5 + 512
        tmp36 = tmp5 < 0
        tmp37 = tl.where(tmp36, tmp35, tmp5)
        tl.device_assert(((0 <= tmp37) & (tmp37 < 512)) | ~xmask, "index out of bounds: 0 <= tmp37 < 512")
        tmp38 = tl.load(in_ptr3 + (r1 + (768*tmp37)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp39 = tmp34 + tmp38
        tmp40 = 768.0
        tmp41 = tmp12 / tmp40
        tmp42 = tmp39 - tmp41
        tmp43 = tmp28 / tmp40
        tmp44 = 1e-07
        tmp45 = tmp43 + tmp44
        tmp46 = tl.sqrt(tmp45)
        tmp47 = tmp42 / tmp46
        tmp48 = tmp30 * tmp47
        tmp50 = tmp48 + tmp49
        tl.store(out_ptr2 + (r1 + (768*x0)), tmp50, rmask & xmask)
