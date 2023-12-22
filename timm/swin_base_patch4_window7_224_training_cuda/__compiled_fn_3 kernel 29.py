
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_28', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x0 = xindex % 256
    x1 = (xindex // 256) % 28
    x2 = (xindex // 7168) % 28
    x3 = (xindex // 200704)
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (256*(((25 + x1) % 28) % 7)) + (1792*(((25 + x2) % 28) % 7)) + (12544*(((25 + x1) % 28) // 7)) + (50176*(((25 + x2) % 28) // 7)) + (200704*x3)), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x3), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (x4), None)
    tmp10 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x3), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = 0.9869565209373832
    tmp6 = tmp4 / tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = tmp12 / tmp5
    tmp14 = tmp11 * tmp13
    tmp15 = tmp8 + tmp14
    tl.store(in_out_ptr0 + (x4), tmp15, None)
