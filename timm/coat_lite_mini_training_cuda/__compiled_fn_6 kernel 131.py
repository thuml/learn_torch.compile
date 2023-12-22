
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_130', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x1 = (xindex // 784)
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp16 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.load(in_ptr3 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = 1 + x0
        tmp1 = tl.full([1, 1], 1, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.load(in_ptr0 + (128 + r2 + (128*x0) + (100480*x1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (784*r2) + (100352*x1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = 0.0
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tmp0 < tmp1
        tmp11 = tl.load(in_ptr0 + (r2 + (100480*x1)), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
        tmp13 = tl.where(tmp10, tmp11, tmp12)
        tmp14 = tl.where(tmp10, tmp13, tmp8)
        tmp15 = tmp9 + tmp14
        tmp17 = tmp15 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
        tmp22 = tmp17 * tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask & xmask, tmp25, _tmp24)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tmp26 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp43 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp48 = tl.load(in_ptr3 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp27 = 1 + x0
        tmp28 = tl.full([1, 1], 1, tl.int64)
        tmp29 = tmp27 >= tmp28
        tmp30 = tl.load(in_ptr0 + (128 + r2 + (128*x0) + (100480*x1)), rmask & tmp29 & xmask, eviction_policy='evict_first', other=0.0)
        tmp31 = tl.load(in_ptr1 + (x0 + (784*r2) + (100352*x1)), rmask & tmp29 & xmask, eviction_policy='evict_first', other=0.0)
        tmp32 = tmp30 + tmp31
        tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
        tmp34 = tl.where(tmp29, tmp32, tmp33)
        tmp35 = 0.0
        tmp36 = tl.where(tmp29, tmp34, tmp35)
        tmp37 = tmp27 < tmp28
        tmp38 = tl.load(in_ptr0 + (r2 + (100480*x1)), rmask & tmp37 & xmask, eviction_policy='evict_last', other=0.0)
        tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
        tmp40 = tl.where(tmp37, tmp38, tmp39)
        tmp41 = tl.where(tmp37, tmp40, tmp35)
        tmp42 = tmp36 + tmp41
        tmp44 = tmp42 * tmp43
        tmp45 = 128.0
        tmp46 = tmp44 * tmp45
        tmp47 = tmp46 - tmp19
        tmp49 = tmp48 * tmp24
        tmp50 = tmp47 - tmp49
        tmp51 = tmp26 * tmp50
        tl.store(out_ptr2 + (r2 + (128*x3)), tmp51, rmask & xmask)
