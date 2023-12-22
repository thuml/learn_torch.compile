
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_addcmul_mean_mul_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 384
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    x4 = (xindex // 384)
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    x1 = (xindex // 384) % 2
    x2 = (xindex // 768)
    tmp9 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp4 = tl.load(in_ptr2 + (x0 + (384*r3) + (37632*x4)), rmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr4 + (r3 + (98*x1) + (196*x0) + (75264*x2)), rmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(in_ptr6 + (x0 + (384*r3) + (37632*x4)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 1.0
        tmp3 = tmp1 * tmp2
        tmp7 = tmp5 * tmp6
        tmp8 = tmp4 + tmp7
        tmp11 = tmp9 * tmp10
        tmp12 = tmp8 + tmp11
        tmp13 = tmp3 * tmp12
        tmp14 = tmp0 + tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask, tmp17, _tmp16)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp16, None)
