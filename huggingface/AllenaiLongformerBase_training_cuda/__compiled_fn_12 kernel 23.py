
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_backward_data_copy_masked_fill_native_dropout_backward_tril_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 513
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 12)
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last').to(tl.int1)
    x0 = xindex % 12
    x3 = xindex
    _tmp31 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp22 = tl.load(in_ptr2 + (r2 + (513*x3)), rmask, eviction_policy='evict_last').to(tl.int1)
        tmp28 = tl.load(in_ptr3 + (r2 + (513*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = r2
        tmp2 = tl.full([1, 1], 770, tl.int64)
        tmp3 = tmp1 < tmp2
        tmp4 = r2 + (770*(x1 % 256))
        tmp5 = tl.full([1, 1], 196864, tl.int64)
        tmp6 = tmp4 < tmp5
        tmp7 = tmp6 & tmp3
        tmp8 = (r2 + (770*(x1 % 256))) % 769
        tmp9 = tl.full([1, 1], 768, tl.int64)
        tmp10 = tmp8 < tmp9
        tmp11 = tmp10 & tmp7
        tmp12 = tl.load(in_ptr1 + ((768*(((r2 + (770*(x1 % 256))) // 769) % 256)) + (196608*(x1 // 256)) + (786432*x0) + ((r2 + (770*(x1 % 256))) % 769)), rmask & tmp11, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
        tmp14 = tl.where(tmp11, tmp12, tmp13)
        tmp15 = 0.0
        tmp16 = tl.where(tmp10, tmp14, tmp15)
        tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
        tmp18 = tl.where(tmp7, tmp16, tmp17)
        tmp19 = tl.where(tmp6, tmp18, tmp15)
        tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
        tmp21 = tl.where(tmp3, tmp19, tmp20)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = 1.1111111111111112
        tmp25 = tmp23 * tmp24
        tmp26 = tmp21 * tmp25
        tmp27 = tl.where(tmp0, tmp15, tmp26)
        tmp29 = tmp27 * tmp28
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
        tmp32 = _tmp31 + tmp30
        _tmp31 = tl.where(rmask, tmp32, _tmp31)
    tmp31 = tl.sum(_tmp31, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp54 = tl.load(in_ptr2 + (r2 + (513*x3)), rmask, eviction_policy='evict_first').to(tl.int1)
        tmp60 = tl.load(in_ptr3 + (r2 + (513*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp33 = r2
        tmp34 = tl.full([1, 1], 770, tl.int64)
        tmp35 = tmp33 < tmp34
        tmp36 = r2 + (770*(x1 % 256))
        tmp37 = tl.full([1, 1], 196864, tl.int64)
        tmp38 = tmp36 < tmp37
        tmp39 = tmp38 & tmp35
        tmp40 = (r2 + (770*(x1 % 256))) % 769
        tmp41 = tl.full([1, 1], 768, tl.int64)
        tmp42 = tmp40 < tmp41
        tmp43 = tmp42 & tmp39
        tmp44 = tl.load(in_ptr1 + ((768*(((r2 + (770*(x1 % 256))) // 769) % 256)) + (196608*(x1 // 256)) + (786432*x0) + ((r2 + (770*(x1 % 256))) % 769)), rmask & tmp43, eviction_policy='evict_last', other=0.0)
        tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
        tmp46 = tl.where(tmp43, tmp44, tmp45)
        tmp47 = 0.0
        tmp48 = tl.where(tmp42, tmp46, tmp47)
        tmp49 = tl.full(tmp48.shape, 0.0, tmp48.dtype)
        tmp50 = tl.where(tmp39, tmp48, tmp49)
        tmp51 = tl.where(tmp38, tmp50, tmp47)
        tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
        tmp53 = tl.where(tmp35, tmp51, tmp52)
        tmp55 = tmp54.to(tl.float32)
        tmp56 = 1.1111111111111112
        tmp57 = tmp55 * tmp56
        tmp58 = tmp53 * tmp57
        tmp59 = tl.where(tmp0, tmp47, tmp58)
        tmp61 = tmp59 * tmp60
        tmp62 = tmp60 * tmp31
        tmp63 = tmp61 - tmp62
        tl.store(out_ptr1 + (r2 + (513*x1) + (525312*x0)), tmp63, rmask)
