
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 2048],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 65536
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 2048
    x1 = (xindex // 2048) % 8
    _tmp32 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (2048*x4)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = r3 + ((-1)*x0)
        tmp2 = tl.full([1, 1], 0, tl.int64)
        tmp3 = tmp1 > tmp2
        tmp4 = tmp3.to(tl.int64)
        tmp5 = tl.full([1, 1], 16, tl.int64)
        tmp6 = tmp4 * tmp5
        tmp7 = tmp6 + tmp2
        tmp8 = tl.abs(tmp1)
        tmp9 = tl.full([1, 1], 8, tl.int64)
        tmp10 = tmp8 < tmp9
        tmp11 = tmp8.to(tl.float32)
        tmp12 = 8.0
        tmp13 = tmp11 / tmp12
        tmp14 = tl.log(tmp13)
        tmp15 = 2.772588722239781
        tmp16 = tmp14 / tmp15
        tmp17 = tmp16 * tmp12
        tmp18 = tmp17.to(tl.int64)
        tmp19 = tmp18 + tmp9
        tmp20 = tl.full([1, 1], 15, tl.int64)
        tmp21 = triton_helpers.minimum(tmp19, tmp20)
        tmp22 = tl.where(tmp10, tmp8, tmp21)
        tmp23 = tmp7 + tmp22
        tmp24 = tmp23 + 32
        tmp25 = tmp23 < 0
        tmp26 = tl.where(tmp25, tmp24, tmp23)
        tl.device_assert(((0 <= tmp26) & (tmp26 < 32)) | ~rmask, "index out of bounds: 0 <= tmp26 < 32")
        tmp27 = tl.load(in_ptr1 + (x1 + (8*tmp26)), rmask, eviction_policy='evict_last', other=0.0)
        tmp28 = 0.0
        tmp29 = tmp27 + tmp28
        tmp30 = tmp0 + tmp29
        tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
        tmp33 = triton_helpers.maximum(_tmp32, tmp31)
        _tmp32 = tl.where(rmask, tmp33, _tmp32)
    tmp32 = triton_helpers.max2(_tmp32, 1)[:, None]
    _tmp68 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp34 = tl.load(in_ptr0 + (r3 + (2048*x4)), rmask, eviction_policy='evict_last', other=0.0)
        tmp35 = r3 + ((-1)*x0)
        tmp36 = tl.full([1, 1], 0, tl.int64)
        tmp37 = tmp35 > tmp36
        tmp38 = tmp37.to(tl.int64)
        tmp39 = tl.full([1, 1], 16, tl.int64)
        tmp40 = tmp38 * tmp39
        tmp41 = tmp40 + tmp36
        tmp42 = tl.abs(tmp35)
        tmp43 = tl.full([1, 1], 8, tl.int64)
        tmp44 = tmp42 < tmp43
        tmp45 = tmp42.to(tl.float32)
        tmp46 = 8.0
        tmp47 = tmp45 / tmp46
        tmp48 = tl.log(tmp47)
        tmp49 = 2.772588722239781
        tmp50 = tmp48 / tmp49
        tmp51 = tmp50 * tmp46
        tmp52 = tmp51.to(tl.int64)
        tmp53 = tmp52 + tmp43
        tmp54 = tl.full([1, 1], 15, tl.int64)
        tmp55 = triton_helpers.minimum(tmp53, tmp54)
        tmp56 = tl.where(tmp44, tmp42, tmp55)
        tmp57 = tmp41 + tmp56
        tmp58 = tmp57 + 32
        tmp59 = tmp57 < 0
        tmp60 = tl.where(tmp59, tmp58, tmp57)
        tl.device_assert(((0 <= tmp60) & (tmp60 < 32)) | ~rmask, "index out of bounds: 0 <= tmp60 < 32")
        tmp61 = tl.load(in_ptr1 + (x1 + (8*tmp60)), rmask, eviction_policy='evict_last', other=0.0)
        tmp62 = 0.0
        tmp63 = tmp61 + tmp62
        tmp64 = tmp34 + tmp63
        tmp65 = tmp64 - tmp32
        tmp66 = tl.exp(tmp65)
        tmp67 = tl.broadcast_to(tmp66, [XBLOCK, RBLOCK])
        tmp69 = _tmp68 + tmp67
        _tmp68 = tl.where(rmask, tmp69, _tmp68)
    tmp68 = tl.sum(_tmp68, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp70 = tl.load(in_ptr0 + (r3 + (2048*x4)), rmask, eviction_policy='evict_first', other=0.0)
        tmp71 = r3 + ((-1)*x0)
        tmp72 = tl.full([1, 1], 0, tl.int64)
        tmp73 = tmp71 > tmp72
        tmp74 = tmp73.to(tl.int64)
        tmp75 = tl.full([1, 1], 16, tl.int64)
        tmp76 = tmp74 * tmp75
        tmp77 = tmp76 + tmp72
        tmp78 = tl.abs(tmp71)
        tmp79 = tl.full([1, 1], 8, tl.int64)
        tmp80 = tmp78 < tmp79
        tmp81 = tmp78.to(tl.float32)
        tmp82 = 8.0
        tmp83 = tmp81 / tmp82
        tmp84 = tl.log(tmp83)
        tmp85 = 2.772588722239781
        tmp86 = tmp84 / tmp85
        tmp87 = tmp86 * tmp82
        tmp88 = tmp87.to(tl.int64)
        tmp89 = tmp88 + tmp79
        tmp90 = tl.full([1, 1], 15, tl.int64)
        tmp91 = triton_helpers.minimum(tmp89, tmp90)
        tmp92 = tl.where(tmp80, tmp78, tmp91)
        tmp93 = tmp77 + tmp92
        tmp94 = tmp93 + 32
        tmp95 = tmp93 < 0
        tmp96 = tl.where(tmp95, tmp94, tmp93)
        tl.device_assert(((0 <= tmp96) & (tmp96 < 32)) | ~rmask, "index out of bounds: 0 <= tmp96 < 32")
        tmp97 = tl.load(in_ptr1 + (x1 + (8*tmp96)), rmask, eviction_policy='evict_last', other=0.0)
        tmp98 = 0.0
        tmp99 = tmp97 + tmp98
        tmp100 = tmp70 + tmp99
        tmp101 = tmp100 - tmp32
        tmp102 = tl.exp(tmp101)
        tmp103 = tmp102 / tmp68
        tl.store(out_ptr2 + (r3 + (2048*x4)), tmp103, rmask)
