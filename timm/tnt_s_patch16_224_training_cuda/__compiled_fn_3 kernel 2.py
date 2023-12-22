
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 4) % 4
    x3 = (xindex // 384)
    x0 = xindex % 4
    x2 = (xindex // 16) % 24
    x6 = xindex % 384
    x7 = xindex
    tmp0 = tl.load(in_ptr0 + ((14*x1) + ((x3 % 196) // 14)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + ((14*x0) + ((x3 % 196) % 14)), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x6), None, eviction_policy='evict_last')
    tmp1 = tmp0 + 56
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert((0 <= tmp3) & (tmp3 < 56), "index out of bounds: 0 <= tmp3 < 56")
    tmp5 = tmp4 + 56
    tmp6 = tmp4 < 0
    tmp7 = tl.where(tmp6, tmp5, tmp4)
    tl.device_assert((0 <= tmp7) & (tmp7 < 56), "index out of bounds: 0 <= tmp7 < 56")
    tmp8 = tl.load(in_ptr1 + (tmp7 + (56*tmp3) + (3136*x2) + (75264*(x3 // 196))), None, eviction_policy='evict_last')
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tl.store(out_ptr0 + (x7), tmp12, None)
