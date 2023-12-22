
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9461760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 770
    x2 = (xindex // 9240)
    x3 = (xindex // 770)
    x1 = (xindex // 770) % 12
    tmp0 = x0
    tmp1 = tl.full([1], 513, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x2), tmp2, eviction_policy='evict_last').to(tl.int1)
    tmp4 = tl.load(in_ptr1 + (x0 + (513*x3)), tmp2, other=0.0)
    tmp5 = tl.load(in_ptr2 + (x3), tmp2, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp4 - tmp5
    tmp7 = tl.exp(tmp6)
    tmp8 = tl.load(in_ptr3 + (x3), tmp2, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 / tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp3, tmp10, tmp9)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp2, tmp11, tmp12)
    tl.store(out_ptr0 + (x0 + (770*x2) + (788480*x1)), tmp13, None)
