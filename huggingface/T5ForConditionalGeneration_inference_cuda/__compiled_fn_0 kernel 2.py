
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr1, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp33 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = (-1)*(tl.math.min(0, r2 + ((-1)*x0)))
        tmp2 = tl.full([1, 1], 16, tl.int64)
        tmp3 = tmp1 < tmp2
        tmp4 = tmp1.to(tl.float32)
        tmp5 = 16.0
        tmp6 = tmp4 / tmp5
        tmp7 = tl.log(tmp6)
        tmp8 = 2.0794415416798357
        tmp9 = tmp7 / tmp8
        tmp10 = tmp9 * tmp5
        tmp11 = tmp10.to(tl.int64)
        tmp12 = tmp11 + tmp2
        tmp13 = tl.full([1, 1], 31, tl.int64)
        tmp14 = triton_helpers.minimum(tmp12, tmp13)
        tmp15 = tl.where(tmp3, tmp1, tmp14)
        tmp16 = tl.full([1, 1], 0, tl.int64)
        tmp17 = tmp15 + tmp16
        tmp18 = tmp17 + 32
        tmp19 = tmp17 < 0
        tmp20 = tl.where(tmp19, tmp18, tmp17)
        tl.device_assert((0 <= tmp20) & (tmp20 < 32), "index out of bounds: 0 <= tmp20 < 32")
        tmp21 = tl.load(in_ptr1 + (x1 + (8*tmp20)), None, eviction_policy='evict_last')
        tmp22 = r2
        tmp23 = x0
        tmp24 = tmp22 <= tmp23
        tmp25 = tmp24.to(tl.float32)
        tmp26 = 1.0
        tmp27 = tmp26 - tmp25
        tmp28 = -3.4028234663852886e+38
        tmp29 = tmp27 * tmp28
        tmp30 = tmp21 + tmp29
        tmp31 = tmp0 + tmp30
        tmp32 = tl.broadcast_to(tmp31, [XBLOCK, RBLOCK])
        tmp34 = triton_helpers.maximum(_tmp33, tmp32)
        _tmp33 = tl.where(rmask, tmp34, _tmp33)
    tmp33 = triton_helpers.max2(_tmp33, 1)[:, None]
    _tmp70 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp35 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp36 = (-1)*(tl.math.min(0, r2 + ((-1)*x0)))
        tmp37 = tl.full([1, 1], 16, tl.int64)
        tmp38 = tmp36 < tmp37
        tmp39 = tmp36.to(tl.float32)
        tmp40 = 16.0
        tmp41 = tmp39 / tmp40
        tmp42 = tl.log(tmp41)
        tmp43 = 2.0794415416798357
        tmp44 = tmp42 / tmp43
        tmp45 = tmp44 * tmp40
        tmp46 = tmp45.to(tl.int64)
        tmp47 = tmp46 + tmp37
        tmp48 = tl.full([1, 1], 31, tl.int64)
        tmp49 = triton_helpers.minimum(tmp47, tmp48)
        tmp50 = tl.where(tmp38, tmp36, tmp49)
        tmp51 = tl.full([1, 1], 0, tl.int64)
        tmp52 = tmp50 + tmp51
        tmp53 = tmp52 + 32
        tmp54 = tmp52 < 0
        tmp55 = tl.where(tmp54, tmp53, tmp52)
        tl.device_assert((0 <= tmp55) & (tmp55 < 32), "index out of bounds: 0 <= tmp55 < 32")
        tmp56 = tl.load(in_ptr1 + (x1 + (8*tmp55)), None, eviction_policy='evict_last')
        tmp57 = r2
        tmp58 = x0
        tmp59 = tmp57 <= tmp58
        tmp60 = tmp59.to(tl.float32)
        tmp61 = 1.0
        tmp62 = tmp61 - tmp60
        tmp63 = -3.4028234663852886e+38
        tmp64 = tmp62 * tmp63
        tmp65 = tmp56 + tmp64
        tmp66 = tmp35 + tmp65
        tmp67 = tmp66 - tmp33
        tmp68 = tl.exp(tmp67)
        tmp69 = tl.broadcast_to(tmp68, [XBLOCK, RBLOCK])
        tmp71 = _tmp70 + tmp69
        _tmp70 = tl.where(rmask, tmp71, _tmp70)
        tl.store(out_ptr1 + (r2 + (1024*x3)), tmp67, rmask)
    tmp70 = tl.sum(_tmp70, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp72 = tl.load(out_ptr1 + (r2 + (1024*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp73 = tl.exp(tmp72)
        tmp74 = tmp73 / tmp70
        tl.store(out_ptr3 + (r2 + (1024*x3)), tmp74, rmask)
