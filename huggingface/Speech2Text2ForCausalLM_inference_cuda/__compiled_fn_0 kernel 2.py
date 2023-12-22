
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_embedding_mul_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 256)
    x0 = xindex % 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp1 = tmp0 + 10000
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert((0 <= tmp3) & (tmp3 < 10000), "index out of bounds: 0 <= tmp3 < 10000")
    tmp4 = tl.load(in_ptr1 + (x0 + (256*tmp3)), None)
    tmp5 = 16.0
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.full([1], 0, tl.int32)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.full([1], 1, tl.int64)
    tmp12 = tmp0 != tmp11
    tmp13 = tmp12.to(tl.int32)
    tmp14 = tmp10 * tmp13
    tmp15 = tmp14.to(tl.int64)
    tmp16 = tmp15 + tmp11
    tmp17 = tmp16 + 1026
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tl.device_assert((0 <= tmp19) & (tmp19 < 1026), "index out of bounds: 0 <= tmp19 < 1026")
    tmp20 = tl.load(in_ptr3 + (x0 + (256*tmp19)), None)
    tmp21 = tmp6 + tmp20
    tl.store(out_ptr0 + (x2), tmp21, None)
