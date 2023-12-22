
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_gelu_backward_mul_sigmoid_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 884736
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 144)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp6 = tl.load(in_ptr2 + (x2), None)
    tmp11 = tl.load(in_ptr3 + (0))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK])
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = 0.9128709291752768
    tmp3 = tmp1 * tmp2
    tmp4 = 1.7015043497085571
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tmp0 + tmp7
    tmp9 = 0.2
    tmp10 = tmp8 * tmp9
    tmp13 = tmp10 * tmp12
    tmp14 = 2.0
    tmp15 = tmp13 * tmp14
    tmp17 = tl.sigmoid(tmp16)
    tmp18 = tmp15 * tmp17
    tmp20 = 144.0
    tmp21 = tmp19 / tmp20
    tmp22 = tmp18 + tmp21
    tl.store(out_ptr0 + (x2), tmp22, None)
