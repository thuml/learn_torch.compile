
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_96', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3328
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 256)
    x0 = xindex % 256
    _tmp32 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp41 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.broadcast_to(x0 % 64, [XBLOCK, RBLOCK])
        tmp4 = tl.full([1, 1], 0, tl.int64)
        tmp5 = tmp3 >= tmp4
        tmp6 = tl.full([1, 1], 16, tl.int64)
        tmp7 = tmp3 < tmp6
        tmp8 = tmp7 & tmp2
        tmp9 = tl.load(in_ptr0 + ((16*((r2 + (121*x1)) % 196)) + (3136*(x0 // 64)) + (12544*(((r2 + (121*x1)) // 196) % 8)) + (x0 % 64)), rmask & tmp8 & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
        tmp11 = tl.where(tmp8, tmp9, tmp10)
        tmp12 = tmp3 >= tmp6
        tmp13 = tl.full([1, 1], 32, tl.int64)
        tmp14 = tmp3 < tmp13
        tmp15 = tmp12 & tmp14
        tmp16 = tmp15 & tmp2
        tmp17 = tl.load(in_ptr1 + ((-3136) + (196*(x0 % 64)) + (3136*(x0 // 64)) + (12544*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp16 & xmask, eviction_policy='evict_last', other=0.0)
        tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
        tmp19 = tl.where(tmp16, tmp17, tmp18)
        tmp20 = tmp3 >= tmp13
        tmp21 = tl.full([1, 1], 64, tl.int64)
        tmp22 = tmp3 < tmp21
        tmp23 = tmp20 & tmp2
        tmp24 = tl.load(in_ptr2 + ((-32) + (32*((r2 + (121*x1)) % 196)) + (6272*(x0 // 64)) + (25088*(((r2 + (121*x1)) // 196) % 8)) + (x0 % 64)), rmask & tmp23 & xmask, eviction_policy='evict_first', other=0.0)
        tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
        tmp26 = tl.where(tmp23, tmp24, tmp25)
        tmp27 = tl.where(tmp15, tmp19, tmp26)
        tmp28 = tl.where(tmp7, tmp11, tmp27)
        tmp29 = tl.full(tmp28.shape, 0, tmp28.dtype)
        tmp30 = tl.where(tmp2, tmp28, tmp29)
        tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
        tmp33 = _tmp32 + tmp31
        _tmp32 = tl.where(rmask & xmask, tmp33, _tmp32)
        tmp34 = tl.load(in_ptr3 + (x0 + (256*r2) + (30976*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp35 = tl.load(in_ptr4 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp36 = tmp34 - tmp35
        tmp37 = tmp28 * tmp36
        tmp38 = tl.full(tmp37.shape, 0, tmp37.dtype)
        tmp39 = tl.where(tmp2, tmp37, tmp38)
        tmp40 = tl.broadcast_to(tmp39, [XBLOCK, RBLOCK])
        tmp42 = _tmp41 + tmp40
        _tmp41 = tl.where(rmask & xmask, tmp42, _tmp41)
    tmp32 = tl.sum(_tmp32, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp32, xmask)
    tmp41 = tl.sum(_tmp41, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp41, xmask)
