
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2709504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 32) % 9
    x2 = (xindex // 288) % 196
    x0 = xindex % 32
    x3 = (xindex // 56448) % 6
    x4 = (xindex // 338688)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + ((14*(x1 // 3)) + (x2 // 14)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + ((14*(x1 % 3)) + (x2 % 14)), None, eviction_policy='evict_last')
    tmp1 = tmp0 + 30
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert((0 <= tmp3) & (tmp3 < 30), "index out of bounds: 0 <= tmp3 < 30")
    tmp5 = tmp4 + 30
    tmp6 = tmp4 < 0
    tmp7 = tl.where(tmp6, tmp5, tmp4)
    tl.device_assert((0 <= tmp7) & (tmp7 < 30), "index out of bounds: 0 <= tmp7 < 30")
    tmp8 = (-1) + tmp3
    tmp9 = tl.full([1], 0, tl.int64)
    tmp10 = tmp8 >= tmp9
    tmp11 = tl.full([1], 28, tl.int64)
    tmp12 = tmp8 < tmp11
    tmp13 = (-1) + tmp7
    tmp14 = tmp13 >= tmp9
    tmp15 = tmp13 < tmp11
    tmp16 = tmp10 & tmp12
    tmp17 = tmp16 & tmp14
    tmp18 = tmp17 & tmp15
    tmp19 = tl.load(in_ptr2 + ((-5568) + x0 + (32*x3) + (192*tmp7) + (5376*tmp3) + (150528*x4)), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tl.store(out_ptr0 + (x6), tmp21, None)
