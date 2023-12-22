
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_leaky_relu_backward_native_batch_norm_backward_132', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    x3 = xindex
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp25 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp29 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (4194304*x1)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp24 = tl.load(in_ptr4 + (x0 + (128*r2) + (4194304*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = x0
        tmp2 = tl.full([1, 1], 0, tl.int64)
        tmp3 = tmp1 >= tmp2
        tmp4 = tl.full([1, 1], 64, tl.int64)
        tmp5 = tmp1 < tmp4
        tmp6 = tl.load(in_ptr1 + ((16384*x0) + (2097152*(r2 // 16384)) + (4194304*x1) + (r2 % 16384)), rmask & tmp5 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
        tmp8 = tl.where(tmp5, tmp6, tmp7)
        tmp9 = tmp1 >= tmp4
        tmp10 = tl.full([1, 1], 128, tl.int64)
        tmp11 = tmp1 < tmp10
        tmp12 = tl.load(in_ptr2 + ((-1048576) + (16384*x3) + (1048576*(r2 // 16384)) + (r2 % 16384)), rmask & tmp9 & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_ptr3 + ((-1048576) + (16384*x3) + (1048576*(r2 // 16384)) + (r2 % 16384)), rmask & tmp9 & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tmp12 + tmp13
        tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
        tmp16 = tl.where(tmp9, tmp14, tmp15)
        tmp17 = tl.where(tmp5, tmp8, tmp16)
        tmp18 = 0.01
        tmp19 = tmp17 * tmp18
        tmp20 = tl.where(tmp0, tmp17, tmp19)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask & xmask, tmp23, _tmp22)
        tmp26 = tmp24 - tmp25
        tmp27 = tmp20 * tmp26
        tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
        tmp30 = _tmp29 + tmp28
        _tmp29 = tl.where(rmask & xmask, tmp30, _tmp29)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp22, xmask)
    tmp29 = tl.sum(_tmp29, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp29, xmask)
