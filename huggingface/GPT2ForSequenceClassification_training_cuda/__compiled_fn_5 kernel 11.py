
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_argmax_eq_sub_10', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp5 = tl.full([XBLOCK, RBLOCK], -9223372036854775808, tl.int64)
    _tmp5_index = tl.full([XBLOCK, RBLOCK], 9223372036854775807, tl.int64)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 == tmp1
        tmp3 = tmp2.to(tl.int64)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        _tmp5_next, _tmp5_index_next = triton_helpers.maximum_with_index(
            _tmp5, _tmp5_index, tmp4, rindex
        )
        _tmp5 = tl.where(rmask, _tmp5_next, _tmp5)
        _tmp5_index = tl.where(rmask, _tmp5_index_next, _tmp5_index)
    _, tmp5_tmp = triton_helpers.max_with_index(_tmp5, _tmp5_index, 1)
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tl.full([1, 1], 1, tl.int64)
    tmp7 = tmp5 - tmp6
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp7, None)
