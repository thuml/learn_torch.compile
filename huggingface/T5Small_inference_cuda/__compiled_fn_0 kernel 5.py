
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp30 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = r2 + ((-1)*x0)
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
        tmp28 = tmp0 + tmp27
        tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
        tmp31 = triton_helpers.maximum(_tmp30, tmp29)
        _tmp30 = tl.where(rmask, tmp31, _tmp30)
    tmp30 = triton_helpers.max2(_tmp30, 1)[:, None]
    _tmp64 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp32 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp33 = r2 + ((-1)*x0)
        tmp34 = tl.full([1, 1], 0, tl.int64)
        tmp35 = tmp33 > tmp34
        tmp36 = tmp35.to(tl.int64)
        tmp37 = tl.full([1, 1], 16, tl.int64)
        tmp38 = tmp36 * tmp37
        tmp39 = tmp38 + tmp34
        tmp40 = tl.abs(tmp33)
        tmp41 = tl.full([1, 1], 8, tl.int64)
        tmp42 = tmp40 < tmp41
        tmp43 = tmp40.to(tl.float32)
        tmp44 = 8.0
        tmp45 = tmp43 / tmp44
        tmp46 = tl.log(tmp45)
        tmp47 = 2.772588722239781
        tmp48 = tmp46 / tmp47
        tmp49 = tmp48 * tmp44
        tmp50 = tmp49.to(tl.int64)
        tmp51 = tmp50 + tmp41
        tmp52 = tl.full([1, 1], 15, tl.int64)
        tmp53 = triton_helpers.minimum(tmp51, tmp52)
        tmp54 = tl.where(tmp42, tmp40, tmp53)
        tmp55 = tmp39 + tmp54
        tmp56 = tmp55 + 32
        tmp57 = tmp55 < 0
        tmp58 = tl.where(tmp57, tmp56, tmp55)
        tl.device_assert(((0 <= tmp58) & (tmp58 < 32)) | ~rmask, "index out of bounds: 0 <= tmp58 < 32")
        tmp59 = tl.load(in_ptr1 + (x1 + (8*tmp58)), rmask, eviction_policy='evict_last', other=0.0)
        tmp60 = tmp32 + tmp59
        tmp61 = tmp60 - tmp30
        tmp62 = tl.exp(tmp61)
        tmp63 = tl.broadcast_to(tmp62, [XBLOCK, RBLOCK])
        tmp65 = _tmp64 + tmp63
        _tmp64 = tl.where(rmask, tmp65, _tmp64)
    tmp64 = tl.sum(_tmp64, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp66 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp67 = r2 + ((-1)*x0)
        tmp68 = tl.full([1, 1], 0, tl.int64)
        tmp69 = tmp67 > tmp68
        tmp70 = tmp69.to(tl.int64)
        tmp71 = tl.full([1, 1], 16, tl.int64)
        tmp72 = tmp70 * tmp71
        tmp73 = tmp72 + tmp68
        tmp74 = tl.abs(tmp67)
        tmp75 = tl.full([1, 1], 8, tl.int64)
        tmp76 = tmp74 < tmp75
        tmp77 = tmp74.to(tl.float32)
        tmp78 = 8.0
        tmp79 = tmp77 / tmp78
        tmp80 = tl.log(tmp79)
        tmp81 = 2.772588722239781
        tmp82 = tmp80 / tmp81
        tmp83 = tmp82 * tmp78
        tmp84 = tmp83.to(tl.int64)
        tmp85 = tmp84 + tmp75
        tmp86 = tl.full([1, 1], 15, tl.int64)
        tmp87 = triton_helpers.minimum(tmp85, tmp86)
        tmp88 = tl.where(tmp76, tmp74, tmp87)
        tmp89 = tmp73 + tmp88
        tmp90 = tmp89 + 32
        tmp91 = tmp89 < 0
        tmp92 = tl.where(tmp91, tmp90, tmp89)
        tl.device_assert(((0 <= tmp92) & (tmp92 < 32)) | ~rmask, "index out of bounds: 0 <= tmp92 < 32")
        tmp93 = tl.load(in_ptr1 + (x1 + (8*tmp92)), rmask, eviction_policy='evict_last', other=0.0)
        tmp94 = tmp66 + tmp93
        tmp95 = tmp94 - tmp30
        tmp96 = tl.exp(tmp95)
        tmp97 = tmp96 / tmp64
        tl.store(out_ptr2 + (r2 + (1024*x3)), tmp97, rmask)
