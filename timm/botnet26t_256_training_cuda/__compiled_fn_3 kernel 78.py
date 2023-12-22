
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_add_mul_77', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp28 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (256*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.08838834764831845
        tmp2 = tmp0 * tmp1
        tmp3 = 15 + (31*(x0 // 16)) + (r2 // 16)
        tmp4 = tl.full([1, 1], 512, tl.int64)
        tmp5 = tmp3 < tmp4
        tmp6 = (15 + (31*(x0 // 16)) + (r2 // 16)) % 32
        tmp7 = tl.full([1, 1], 31, tl.int64)
        tmp8 = tmp6 < tmp7
        tmp9 = tmp8 & tmp5
        tmp10 = tl.load(in_ptr1 + ((31*((15 + (31*(x0 // 16)) + (r2 // 16)) // 32)) + (496*(x0 % 16)) + (7936*x1) + ((15 + (31*(x0 // 16)) + (r2 // 16)) % 32)), rmask & tmp9, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
        tmp12 = tl.where(tmp9, tmp10, tmp11)
        tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
        tmp14 = tl.where(tmp5, tmp12, tmp13)
        tmp15 = 15 + (31*(x0 % 16)) + (r2 % 16)
        tmp16 = tmp15 < tmp4
        tmp17 = (15 + (31*(x0 % 16)) + (r2 % 16)) % 32
        tmp18 = tmp17 < tmp7
        tmp19 = tmp18 & tmp16
        tmp20 = tl.load(in_ptr2 + ((31*(((15 + (31*(x0 % 16)) + (r2 % 16)) // 32) % 16)) + (496*(x0 // 16)) + (7936*x1) + ((15 + (31*(x0 % 16)) + (r2 % 16)) % 32)), rmask & tmp19, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
        tmp22 = tl.where(tmp19, tmp20, tmp21)
        tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
        tmp24 = tl.where(tmp16, tmp22, tmp23)
        tmp25 = tmp14 + tmp24
        tmp26 = tmp2 + tmp25
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp29 = triton_helpers.maximum(_tmp28, tmp27)
        _tmp28 = tl.where(rmask, tmp29, _tmp28)
    tmp28 = triton_helpers.max2(_tmp28, 1)[:, None]
    _tmp60 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp30 = tl.load(in_ptr0 + (r2 + (256*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp31 = 0.08838834764831845
        tmp32 = tmp30 * tmp31
        tmp33 = 15 + (31*(x0 // 16)) + (r2 // 16)
        tmp34 = tl.full([1, 1], 512, tl.int64)
        tmp35 = tmp33 < tmp34
        tmp36 = (15 + (31*(x0 // 16)) + (r2 // 16)) % 32
        tmp37 = tl.full([1, 1], 31, tl.int64)
        tmp38 = tmp36 < tmp37
        tmp39 = tmp38 & tmp35
        tmp40 = tl.load(in_ptr1 + ((31*((15 + (31*(x0 // 16)) + (r2 // 16)) // 32)) + (496*(x0 % 16)) + (7936*x1) + ((15 + (31*(x0 // 16)) + (r2 // 16)) % 32)), rmask & tmp39, eviction_policy='evict_last', other=0.0)
        tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
        tmp42 = tl.where(tmp39, tmp40, tmp41)
        tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
        tmp44 = tl.where(tmp35, tmp42, tmp43)
        tmp45 = 15 + (31*(x0 % 16)) + (r2 % 16)
        tmp46 = tmp45 < tmp34
        tmp47 = (15 + (31*(x0 % 16)) + (r2 % 16)) % 32
        tmp48 = tmp47 < tmp37
        tmp49 = tmp48 & tmp46
        tmp50 = tl.load(in_ptr2 + ((31*(((15 + (31*(x0 % 16)) + (r2 % 16)) // 32) % 16)) + (496*(x0 // 16)) + (7936*x1) + ((15 + (31*(x0 % 16)) + (r2 % 16)) % 32)), rmask & tmp49, eviction_policy='evict_last', other=0.0)
        tmp51 = tl.full(tmp50.shape, 0.0, tmp50.dtype)
        tmp52 = tl.where(tmp49, tmp50, tmp51)
        tmp53 = tl.full(tmp52.shape, 0.0, tmp52.dtype)
        tmp54 = tl.where(tmp46, tmp52, tmp53)
        tmp55 = tmp44 + tmp54
        tmp56 = tmp32 + tmp55
        tmp57 = tmp56 - tmp28
        tmp58 = tl.exp(tmp57)
        tmp59 = tl.broadcast_to(tmp58, [XBLOCK, RBLOCK])
        tmp61 = _tmp60 + tmp59
        _tmp60 = tl.where(rmask, tmp61, _tmp60)
    tmp60 = tl.sum(_tmp60, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp62 = tl.load(in_ptr0 + (r2 + (256*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp63 = 0.08838834764831845
        tmp64 = tmp62 * tmp63
        tmp65 = 15 + (31*(x0 // 16)) + (r2 // 16)
        tmp66 = tl.full([1, 1], 512, tl.int64)
        tmp67 = tmp65 < tmp66
        tmp68 = (15 + (31*(x0 // 16)) + (r2 // 16)) % 32
        tmp69 = tl.full([1, 1], 31, tl.int64)
        tmp70 = tmp68 < tmp69
        tmp71 = tmp70 & tmp67
        tmp72 = tl.load(in_ptr1 + ((31*((15 + (31*(x0 // 16)) + (r2 // 16)) // 32)) + (496*(x0 % 16)) + (7936*x1) + ((15 + (31*(x0 // 16)) + (r2 // 16)) % 32)), rmask & tmp71, eviction_policy='evict_last', other=0.0)
        tmp73 = tl.full(tmp72.shape, 0.0, tmp72.dtype)
        tmp74 = tl.where(tmp71, tmp72, tmp73)
        tmp75 = tl.full(tmp74.shape, 0.0, tmp74.dtype)
        tmp76 = tl.where(tmp67, tmp74, tmp75)
        tmp77 = 15 + (31*(x0 % 16)) + (r2 % 16)
        tmp78 = tmp77 < tmp66
        tmp79 = (15 + (31*(x0 % 16)) + (r2 % 16)) % 32
        tmp80 = tmp79 < tmp69
        tmp81 = tmp80 & tmp78
        tmp82 = tl.load(in_ptr2 + ((31*(((15 + (31*(x0 % 16)) + (r2 % 16)) // 32) % 16)) + (496*(x0 // 16)) + (7936*x1) + ((15 + (31*(x0 % 16)) + (r2 % 16)) % 32)), rmask & tmp81, eviction_policy='evict_last', other=0.0)
        tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
        tmp84 = tl.where(tmp81, tmp82, tmp83)
        tmp85 = tl.full(tmp84.shape, 0.0, tmp84.dtype)
        tmp86 = tl.where(tmp78, tmp84, tmp85)
        tmp87 = tmp76 + tmp86
        tmp88 = tmp64 + tmp87
        tmp89 = tmp88 - tmp28
        tmp90 = tl.exp(tmp89)
        tmp91 = tmp90 / tmp60
        tl.store(out_ptr2 + (r2 + (256*x3)), tmp91, rmask)
