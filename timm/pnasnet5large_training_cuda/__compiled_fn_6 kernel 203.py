
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 256],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_constant_pad_nd_native_batch_norm_backward_threshold_backward_202', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 53460
    rnumel = 220
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 990
    x1 = (xindex // 990)
    _tmp34 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((27225*x1) + (1470150*((r2 + (220*x0)) // 27225)) + ((r2 + (220*x0)) % 27225)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr2 + (x1 + (54*r2) + (11880*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = 1 + (((r2 + (220*x0)) // 165) % 165)
        tmp2 = tl.full([1, 1], 0, tl.int64)
        tmp3 = tmp1 >= tmp2
        tmp4 = tl.full([1, 1], 167, tl.int64)
        tmp5 = tmp1 < tmp4
        tmp6 = 1 + ((r2 + (220*x0)) % 165)
        tmp7 = tmp6 >= tmp2
        tmp8 = tmp6 < tmp4
        tmp9 = tmp3 & tmp5
        tmp10 = tmp9 & tmp7
        tmp11 = tmp10 & tmp8
        tmp12 = tl.load(in_ptr1 + (9072 + x1 + (54*((r2 + (220*x0)) % 165)) + (9018*(((r2 + (220*x0)) // 165) % 165)) + (1506006*((r2 + (220*x0)) // 27225))), rmask & tmp11 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
        tmp14 = tl.where(tmp11, tmp12, tmp13)
        tmp15 = tmp0 + tmp14
        tmp17 = 3 + (((r2 + (220*x0)) // 165) % 165)
        tmp18 = tmp17 >= tmp2
        tmp19 = tl.full([1, 1], 171, tl.int64)
        tmp20 = tmp17 < tmp19
        tmp21 = 3 + ((r2 + (220*x0)) % 165)
        tmp22 = tmp21 >= tmp2
        tmp23 = tmp21 < tmp19
        tmp24 = tmp18 & tmp20
        tmp25 = tmp24 & tmp22
        tmp26 = tmp25 & tmp23
        tmp27 = tl.load(in_ptr3 + (516 + (171*(((r2 + (220*x0)) // 165) % 165)) + (29241*x1) + (1579014*((r2 + (220*x0)) // 27225)) + ((r2 + (220*x0)) % 165)), rmask & tmp26 & xmask, eviction_policy='evict_last', other=0.0)
        tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
        tmp29 = tl.where(tmp26, tmp27, tmp28)
        tmp30 = 0.0
        tmp31 = tl.where(tmp16, tmp30, tmp29)
        tmp32 = tmp15 + tmp31
        tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
        tmp35 = _tmp34 + tmp33
        _tmp34 = tl.where(rmask & xmask, tmp35, _tmp34)
    tmp34 = tl.sum(_tmp34, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp34, xmask)
