
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 65536], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_add_mul_rsub_sigmoid_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 38416
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 16
    x5 = xindex
    y4 = yindex
    x3 = (xindex // 196)
    y1 = (yindex // 16)
    tmp0 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_out_ptr0 + (x5 + (38416*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (x3 + (196*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x3 + (196*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0 + (16*x5) + (614656*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (y0 + (16*x3) + (3136*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (y0 + (16*x3) + (3136*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = 1.0
    tmp3 = tmp2 - tmp1
    tmp5 = 0.14433756729740643
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 - tmp7
    tmp9 = tl.exp(tmp8)
    tmp11 = tmp9 / tmp10
    tmp12 = tmp3 * tmp11
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 - tmp16
    tmp18 = tl.exp(tmp17)
    tmp20 = tmp18 / tmp19
    tmp21 = tmp1 * tmp20
    tmp22 = tmp12 + tmp21
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x5 + (38416*y4)), tmp22, xmask & ymask)
