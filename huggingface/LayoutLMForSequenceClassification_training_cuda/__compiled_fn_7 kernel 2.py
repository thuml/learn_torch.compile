
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_forward_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.full([1], -100, tl.int64)
    tmp3 = tmp1 != tmp2
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tl.where(tmp3, tmp1, tmp4)
    tmp6 = tmp5 + 2
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tl.device_assert((0 <= tmp8) & (tmp8 < 2), "index out of bounds: 0 <= tmp8 < 2")
    tmp9 = tl.load(in_ptr1 + (tmp8), None, eviction_policy='evict_last')
    tmp10 = -tmp9
    tmp11 = 0.0
    tmp12 = tl.where(tmp3, tmp10, tmp11)
    tmp13 = tmp3.to(tl.int64)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 / tmp14
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp3, None)
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), tmp15, None)
