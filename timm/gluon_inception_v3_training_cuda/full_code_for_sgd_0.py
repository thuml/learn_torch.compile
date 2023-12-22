
# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def __resume_at_114_8(___stack0):
    foreach = False
    if foreach:
        if torch.jit.is_scripting():
            raise RuntimeError(
                'torch.jit.script not supported with foreach optimizers')
    if foreach:
        if not torch.jit.is_scripting():
            func = _multi_tensor_sgd
        else:
            func = _single_tensor_sgd
    else:
        func = _single_tensor_sgd
    func(params, d_p_list, momentum_buffer_list, weight_decay=weight_decay,
        momentum=momentum, lr=lr, dampening=dampening, nesterov=nesterov,
        has_sparse_grad=has_sparse_grad, maximize=maximize)
    return None

def transformed___resume_at_114_8(___stack0):
    L = {"___stack0": ___stack0}
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return __resume_at_114_8(___stack0)

#============ end of __resume_at_114_8 ============#

def __guard_0_for_sgd(L):
    return (___guarded_code.valid) \
        and (___check_global_state()) \
        and (___check_type_id(L['lr'], 7644160)) \
        and (L['lr'] == 0.01) \
        and (___check_type_id(L['params'], 7642176)) \
        and (len(L['params']) == 284) \
        and (___check_obj_id(L['foreach'], 7677664)) \
        and (___check_type_id(L['d_p_list'], 7642176)) \
        and (len(L['d_p_list']) == 284) \
        and (___check_obj_id(L['maximize'], 7677632)) \
        and (___check_type_id(L['momentum'], 7640416)) \
        and (L['momentum'] == 0) \
        and (___check_obj_id(L['nesterov'], 7677632)) \
        and (___check_type_id(L['dampening'], 7640416)) \
        and (L['dampening'] == 0) \
        and (hasattr(L['d_p_list'][0], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][1], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][2], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][3], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][4], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][5], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][6], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][7], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][8], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][9], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][10], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][11], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][12], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][13], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][14], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][15], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][16], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][17], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][18], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][19], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][20], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][21], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][22], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][23], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][24], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][25], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][26], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][27], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][28], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][29], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][30], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][31], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][32], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][33], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][34], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][35], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][36], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][37], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][38], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][39], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][40], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][41], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][42], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][43], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][44], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][45], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][46], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][47], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][48], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][49], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][50], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][51], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][52], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][53], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][54], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][55], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][56], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][57], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][58], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][59], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][60], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][61], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][62], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][63], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][64], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][65], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][66], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][67], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][68], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][69], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][70], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][71], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][72], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][73], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][74], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][75], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][76], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][77], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][78], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][79], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][80], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][81], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][82], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][83], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][84], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][85], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][86], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][87], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][88], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][89], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][90], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][91], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][92], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][93], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][94], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][95], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][96], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][97], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][98], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][99], '_dynamo_dynamic_indices') == False) \
        and (___check_type_id(L['weight_decay'], 7640416)) \
        and (L['weight_decay'] == 0) \
        and (hasattr(L['d_p_list'][100], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][101], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][102], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][103], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][104], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][105], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][106], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][107], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][108], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][109], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][110], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][111], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][112], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][113], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][114], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][115], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][116], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][117], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][118], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][119], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][120], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][121], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][122], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][123], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][124], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][125], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][126], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][127], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][128], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][129], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][130], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][131], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][132], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][133], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][134], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][135], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][136], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][137], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][138], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][139], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][140], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][141], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][142], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][143], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][144], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][145], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][146], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][147], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][148], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][149], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][150], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][151], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][152], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][153], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][154], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][155], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][156], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][157], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][158], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][159], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][160], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][161], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][162], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][163], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][164], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][165], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][166], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][167], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][168], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][169], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][170], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][171], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][172], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][173], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][174], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][175], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][176], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][177], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][178], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][179], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][180], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][181], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][182], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][183], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][184], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][185], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][186], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][187], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][188], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][189], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][190], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][191], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][192], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][193], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][194], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][195], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][196], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][197], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][198], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][199], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][200], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][201], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][202], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][203], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][204], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][205], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][206], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][207], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][208], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][209], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][210], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][211], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][212], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][213], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][214], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][215], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][216], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][217], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][218], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][219], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][220], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][221], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][222], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][223], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][224], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][225], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][226], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][227], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][228], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][229], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][230], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][231], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][232], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][233], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][234], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][235], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][236], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][237], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][238], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][239], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][240], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][241], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][242], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][243], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][244], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][245], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][246], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][247], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][248], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][249], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][250], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][251], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][252], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][253], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][254], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][255], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][256], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][257], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][258], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][259], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][260], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][261], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][262], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][263], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][264], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][265], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][266], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][267], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][268], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][269], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][270], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][271], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][272], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][273], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][274], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][275], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][276], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][277], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][278], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][279], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][280], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][281], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][282], '_dynamo_dynamic_indices') == False) \
        and (hasattr(L['d_p_list'][283], '_dynamo_dynamic_indices') == False) \
        and (___check_obj_id(L['has_sparse_grad'], 7677632)) \
        and (___check_type_id(L['momentum_buffer_list'], 7642176)) \
        and (len(L['momentum_buffer_list']) == 284) \
        and (___check_obj_id(L['momentum_buffer_list'][0], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][1], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][2], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][3], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][4], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][5], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][6], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][7], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][8], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][9], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][10], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][11], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][12], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][13], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][14], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][15], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][16], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][17], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][18], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][19], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][20], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][21], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][22], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][23], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][24], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][25], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][26], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][27], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][28], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][29], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][30], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][31], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][32], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][33], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][34], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][35], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][36], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][37], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][38], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][39], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][40], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][41], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][42], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][43], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][44], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][45], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][46], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][47], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][48], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][49], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][50], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][51], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][52], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][53], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][54], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][55], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][56], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][57], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][58], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][59], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][60], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][61], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][62], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][63], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][64], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][65], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][66], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][67], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][68], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][69], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][70], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][71], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][72], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][73], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][74], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][75], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][76], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][77], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][78], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][79], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][80], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][81], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][82], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][83], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][84], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][85], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][86], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][87], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][88], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][89], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][90], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][91], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][92], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][93], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][94], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][95], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][96], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][97], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][98], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][99], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][100], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][101], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][102], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][103], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][104], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][105], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][106], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][107], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][108], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][109], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][110], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][111], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][112], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][113], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][114], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][115], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][116], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][117], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][118], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][119], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][120], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][121], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][122], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][123], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][124], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][125], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][126], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][127], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][128], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][129], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][130], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][131], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][132], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][133], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][134], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][135], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][136], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][137], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][138], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][139], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][140], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][141], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][142], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][143], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][144], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][145], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][146], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][147], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][148], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][149], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][150], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][151], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][152], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][153], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][154], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][155], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][156], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][157], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][158], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][159], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][160], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][161], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][162], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][163], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][164], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][165], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][166], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][167], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][168], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][169], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][170], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][171], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][172], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][173], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][174], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][175], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][176], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][177], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][178], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][179], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][180], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][181], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][182], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][183], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][184], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][185], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][186], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][187], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][188], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][189], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][190], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][191], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][192], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][193], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][194], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][195], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][196], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][197], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][198], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][199], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][200], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][201], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][202], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][203], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][204], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][205], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][206], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][207], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][208], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][209], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][210], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][211], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][212], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][213], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][214], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][215], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][216], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][217], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][218], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][219], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][220], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][221], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][222], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][223], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][224], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][225], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][226], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][227], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][228], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][229], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][230], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][231], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][232], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][233], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][234], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][235], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][236], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][237], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][238], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][239], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][240], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][241], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][242], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][243], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][244], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][245], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][246], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][247], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][248], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][249], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][250], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][251], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][252], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][253], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][254], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][255], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][256], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][257], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][258], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][259], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][260], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][261], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][262], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][263], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][264], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][265], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][266], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][267], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][268], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][269], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][270], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][271], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][272], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][273], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][274], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][275], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][276], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][277], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][278], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][279], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][280], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][281], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][282], 7628576)) \
        and (___check_obj_id(L['momentum_buffer_list'][283], 7628576)) \
        and (utils_device.CURRENT_DEVICE == None) \
        and ((___skip_backend_check() or ___current_backend() == ___lookup_backend(139768116722368))) \
        and (___compile_config_hash() == 'a332769668a3739cb7dd5366ab2e983f') \
        and (not ___needs_nopython()) \
        and (___check_tensors(L['params'][0], L['params'][1], L['params'][2], L['params'][3], L['params'][4], L['params'][5], L['params'][6], L['params'][7], L['params'][8], L['params'][9], L['params'][10], L['params'][11], L['params'][12], L['params'][13], L['params'][14], L['params'][15], L['params'][16], L['params'][17], L['params'][18], L['params'][19], L['params'][20], L['params'][21], L['params'][22], L['params'][23], L['params'][24], L['params'][25], L['params'][26], L['params'][27], L['params'][28], L['params'][29], L['params'][30], L['params'][31], L['params'][32], L['params'][33], L['params'][34], L['params'][35], L['params'][36], L['params'][37], L['params'][38], L['params'][39], L['params'][40], L['params'][41], L['params'][42], L['params'][43], L['params'][44], L['params'][45], L['params'][46], L['params'][47], L['params'][48], L['params'][49], L['params'][50], L['params'][51], L['params'][52], L['params'][53], L['params'][54], L['params'][55], L['params'][56], L['params'][57], L['params'][58], L['params'][59], L['params'][60], L['params'][61], L['params'][62], L['params'][63], L['params'][64], L['params'][65], L['params'][66], L['params'][67], L['params'][68], L['params'][69], L['params'][70], L['params'][71], L['params'][72], L['params'][73], L['params'][74], L['params'][75], L['params'][76], L['params'][77], L['params'][78], L['params'][79], L['params'][80], L['params'][81], L['params'][82], L['params'][83], L['params'][84], L['params'][85], L['params'][86], L['params'][87], L['params'][88], L['params'][89], L['params'][90], L['params'][91], L['params'][92], L['params'][93], L['params'][94], L['params'][95], L['params'][96], L['params'][97], L['params'][98], L['params'][99], L['d_p_list'][0], L['d_p_list'][1], L['d_p_list'][2], L['d_p_list'][3], L['d_p_list'][4], L['d_p_list'][5], L['d_p_list'][6], L['d_p_list'][7], L['d_p_list'][8], L['d_p_list'][9], L['params'][100], L['params'][101], L['params'][102], L['params'][103], L['params'][104], L['params'][105], L['params'][106], L['params'][107], L['params'][108], L['params'][109], L['params'][110], L['params'][111], L['params'][112], L['params'][113], L['params'][114], L['params'][115], L['params'][116], L['params'][117], L['params'][118], L['params'][119], L['params'][120], L['params'][121], L['params'][122], L['params'][123], L['params'][124], L['params'][125], L['params'][126], L['params'][127], L['params'][128], L['params'][129], L['params'][130], L['params'][131], L['params'][132], L['params'][133], L['params'][134], L['params'][135], L['params'][136], L['params'][137], L['params'][138], L['params'][139], L['params'][140], L['params'][141], L['params'][142], L['params'][143], L['params'][144], L['params'][145], L['params'][146], L['params'][147], L['params'][148], L['params'][149], L['params'][150], L['params'][151], L['params'][152], L['params'][153], L['params'][154], L['params'][155], L['params'][156], L['params'][157], L['params'][158], L['params'][159], L['params'][160], L['params'][161], L['params'][162], L['params'][163], L['params'][164], L['params'][165], L['params'][166], L['params'][167], L['params'][168], L['params'][169], L['params'][170], L['params'][171], L['params'][172], L['params'][173], L['params'][174], L['params'][175], L['params'][176], L['params'][177], L['params'][178], L['params'][179], L['params'][180], L['params'][181], L['params'][182], L['params'][183], L['params'][184], L['params'][185], L['params'][186], L['params'][187], L['params'][188], L['params'][189], L['params'][190], L['params'][191], L['params'][192], L['params'][193], L['params'][194], L['params'][195], L['params'][196], L['params'][197], L['params'][198], L['params'][199], L['params'][200], L['params'][201], L['params'][202], L['params'][203], L['params'][204], L['params'][205], L['params'][206], L['params'][207], L['params'][208], L['params'][209], L['params'][210], L['params'][211], L['params'][212], L['params'][213], L['params'][214], L['params'][215], L['params'][216], L['params'][217], L['params'][218], L['params'][219], L['params'][220], L['params'][221], L['params'][222], L['params'][223], L['params'][224], L['params'][225], L['params'][226], L['params'][227], L['params'][228], L['params'][229], L['params'][230], L['params'][231], L['params'][232], L['params'][233], L['params'][234], L['params'][235], L['params'][236], L['params'][237], L['params'][238], L['params'][239], L['params'][240], L['params'][241], L['params'][242], L['params'][243], L['params'][244], L['params'][245], L['params'][246], L['params'][247], L['params'][248], L['params'][249], L['params'][250], L['params'][251], L['params'][252], L['params'][253], L['params'][254], L['params'][255], L['params'][256], L['params'][257], L['params'][258], L['params'][259], L['params'][260], L['params'][261], L['params'][262], L['params'][263], L['params'][264], L['params'][265], L['params'][266], L['params'][267], L['params'][268], L['params'][269], L['params'][270], L['params'][271], L['params'][272], L['params'][273], L['params'][274], L['params'][275], L['params'][276], L['params'][277], L['params'][278], L['params'][279], L['params'][280], L['params'][281], L['params'][282], L['params'][283], L['d_p_list'][10], L['d_p_list'][11], L['d_p_list'][12], L['d_p_list'][13], L['d_p_list'][14], L['d_p_list'][15], L['d_p_list'][16], L['d_p_list'][17], L['d_p_list'][18], L['d_p_list'][19], L['d_p_list'][20], L['d_p_list'][21], L['d_p_list'][22], L['d_p_list'][23], L['d_p_list'][24], L['d_p_list'][25], L['d_p_list'][26], L['d_p_list'][27], L['d_p_list'][28], L['d_p_list'][29], L['d_p_list'][30], L['d_p_list'][31], L['d_p_list'][32], L['d_p_list'][33], L['d_p_list'][34], L['d_p_list'][35], L['d_p_list'][36], L['d_p_list'][37], L['d_p_list'][38], L['d_p_list'][39], L['d_p_list'][40], L['d_p_list'][41], L['d_p_list'][42], L['d_p_list'][43], L['d_p_list'][44], L['d_p_list'][45], L['d_p_list'][46], L['d_p_list'][47], L['d_p_list'][48], L['d_p_list'][49], L['d_p_list'][50], L['d_p_list'][51], L['d_p_list'][52], L['d_p_list'][53], L['d_p_list'][54], L['d_p_list'][55], L['d_p_list'][56], L['d_p_list'][57], L['d_p_list'][58], L['d_p_list'][59], L['d_p_list'][60], L['d_p_list'][61], L['d_p_list'][62], L['d_p_list'][63], L['d_p_list'][64], L['d_p_list'][65], L['d_p_list'][66], L['d_p_list'][67], L['d_p_list'][68], L['d_p_list'][69], L['d_p_list'][70], L['d_p_list'][71], L['d_p_list'][72], L['d_p_list'][73], L['d_p_list'][74], L['d_p_list'][75], L['d_p_list'][76], L['d_p_list'][77], L['d_p_list'][78], L['d_p_list'][79], L['d_p_list'][80], L['d_p_list'][81], L['d_p_list'][82], L['d_p_list'][83], L['d_p_list'][84], L['d_p_list'][85], L['d_p_list'][86], L['d_p_list'][87], L['d_p_list'][88], L['d_p_list'][89], L['d_p_list'][90], L['d_p_list'][91], L['d_p_list'][92], L['d_p_list'][93], L['d_p_list'][94], L['d_p_list'][95], L['d_p_list'][96], L['d_p_list'][97], L['d_p_list'][98], L['d_p_list'][99], L['d_p_list'][100], L['d_p_list'][101], L['d_p_list'][102], L['d_p_list'][103], L['d_p_list'][104], L['d_p_list'][105], L['d_p_list'][106], L['d_p_list'][107], L['d_p_list'][108], L['d_p_list'][109], L['d_p_list'][110], L['d_p_list'][111], L['d_p_list'][112], L['d_p_list'][113], L['d_p_list'][114], L['d_p_list'][115], L['d_p_list'][116], L['d_p_list'][117], L['d_p_list'][118], L['d_p_list'][119], L['d_p_list'][120], L['d_p_list'][121], L['d_p_list'][122], L['d_p_list'][123], L['d_p_list'][124], L['d_p_list'][125], L['d_p_list'][126], L['d_p_list'][127], L['d_p_list'][128], L['d_p_list'][129], L['d_p_list'][130], L['d_p_list'][131], L['d_p_list'][132], L['d_p_list'][133], L['d_p_list'][134], L['d_p_list'][135], L['d_p_list'][136], L['d_p_list'][137], L['d_p_list'][138], L['d_p_list'][139], L['d_p_list'][140], L['d_p_list'][141], L['d_p_list'][142], L['d_p_list'][143], L['d_p_list'][144], L['d_p_list'][145], L['d_p_list'][146], L['d_p_list'][147], L['d_p_list'][148], L['d_p_list'][149], L['d_p_list'][150], L['d_p_list'][151], L['d_p_list'][152], L['d_p_list'][153], L['d_p_list'][154], L['d_p_list'][155], L['d_p_list'][156], L['d_p_list'][157], L['d_p_list'][158], L['d_p_list'][159], L['d_p_list'][160], L['d_p_list'][161], L['d_p_list'][162], L['d_p_list'][163], L['d_p_list'][164], L['d_p_list'][165], L['d_p_list'][166], L['d_p_list'][167], L['d_p_list'][168], L['d_p_list'][169], L['d_p_list'][170], L['d_p_list'][171], L['d_p_list'][172], L['d_p_list'][173], L['d_p_list'][174], L['d_p_list'][175], L['d_p_list'][176], L['d_p_list'][177], L['d_p_list'][178], L['d_p_list'][179], L['d_p_list'][180], L['d_p_list'][181], L['d_p_list'][182], L['d_p_list'][183], L['d_p_list'][184], L['d_p_list'][185], L['d_p_list'][186], L['d_p_list'][187], L['d_p_list'][188], L['d_p_list'][189], L['d_p_list'][190], L['d_p_list'][191], L['d_p_list'][192], L['d_p_list'][193], L['d_p_list'][194], L['d_p_list'][195], L['d_p_list'][196], L['d_p_list'][197], L['d_p_list'][198], L['d_p_list'][199], L['d_p_list'][200], L['d_p_list'][201], L['d_p_list'][202], L['d_p_list'][203], L['d_p_list'][204], L['d_p_list'][205], L['d_p_list'][206], L['d_p_list'][207], L['d_p_list'][208], L['d_p_list'][209], L['d_p_list'][210], L['d_p_list'][211], L['d_p_list'][212], L['d_p_list'][213], L['d_p_list'][214], L['d_p_list'][215], L['d_p_list'][216], L['d_p_list'][217], L['d_p_list'][218], L['d_p_list'][219], L['d_p_list'][220], L['d_p_list'][221], L['d_p_list'][222], L['d_p_list'][223], L['d_p_list'][224], L['d_p_list'][225], L['d_p_list'][226], L['d_p_list'][227], L['d_p_list'][228], L['d_p_list'][229], L['d_p_list'][230], L['d_p_list'][231], L['d_p_list'][232], L['d_p_list'][233], L['d_p_list'][234], L['d_p_list'][235], L['d_p_list'][236], L['d_p_list'][237], L['d_p_list'][238], L['d_p_list'][239], L['d_p_list'][240], L['d_p_list'][241], L['d_p_list'][242], L['d_p_list'][243], L['d_p_list'][244], L['d_p_list'][245], L['d_p_list'][246], L['d_p_list'][247], L['d_p_list'][248], L['d_p_list'][249], L['d_p_list'][250], L['d_p_list'][251], L['d_p_list'][252], L['d_p_list'][253], L['d_p_list'][254], L['d_p_list'][255], L['d_p_list'][256], L['d_p_list'][257], L['d_p_list'][258], L['d_p_list'][259], L['d_p_list'][260], L['d_p_list'][261], L['d_p_list'][262], L['d_p_list'][263], L['d_p_list'][264], L['d_p_list'][265], L['d_p_list'][266], L['d_p_list'][267], L['d_p_list'][268], L['d_p_list'][269], L['d_p_list'][270], L['d_p_list'][271], L['d_p_list'][272], L['d_p_list'][273], L['d_p_list'][274], L['d_p_list'][275], L['d_p_list'][276], L['d_p_list'][277], L['d_p_list'][278], L['d_p_list'][279], L['d_p_list'][280], L['d_p_list'][281], L['d_p_list'][282], L['d_p_list'][283], tensor_check_names=tensor_check_names))

def __transformed_code_0_for_sgd(params, d_p_list, momentum_buffer_list, has_sparse_grad, foreach, weight_decay, momentum, lr, dampening, nesterov, maximize):
    _ = None; func = None # this line helps the compiler to generate bytecode with at least the same number of local variables as the original function
    return __resume_at_114_8(_multi_tensor_sgd(params, d_p_list,
        momentum_buffer_list, weight_decay=weight_decay, momentum=momentum, lr=
        lr, dampening=dampening, nesterov=nesterov, has_sparse_grad=
        has_sparse_grad, maximize=maximize))


# Note: if there is a transformed version below, this function might well not be executed directly. Please check the transformed version if possible.
def sgd(params, d_p_list, momentum_buffer_list, has_sparse_grad, foreach, weight_decay, momentum, lr, dampening, nesterov, maximize):
    if foreach is None:
        if not torch.jit.is_scripting():
            __temp_84 = _default_to_fused_or_foreach(params, differentiable=
                False, use_fused=False)
            _ = __temp_84[0]
            foreach = __temp_84[1]
        else:
            foreach = False
    if foreach:
        if torch.jit.is_scripting():
            raise RuntimeError(
                'torch.jit.script not supported with foreach optimizers')
    if foreach:
        if not torch.jit.is_scripting():
            func = _multi_tensor_sgd
        else:
            func = _single_tensor_sgd
    else:
        func = _single_tensor_sgd
    func(params, d_p_list, momentum_buffer_list, weight_decay=weight_decay,
        momentum=momentum, lr=lr, dampening=dampening, nesterov=nesterov,
        has_sparse_grad=has_sparse_grad, maximize=maximize)
    return None

def transformed_sgd(params, d_p_list, momentum_buffer_list, has_sparse_grad, foreach, weight_decay, momentum, lr, dampening, nesterov, maximize):
    L = {"params": params, "d_p_list": d_p_list, "momentum_buffer_list": momentum_buffer_list, "has_sparse_grad": has_sparse_grad, "foreach": foreach, "weight_decay": weight_decay, "momentum": momentum, "lr": lr, "dampening": dampening, "nesterov": nesterov, "maximize": maximize}
    if __guard_0_for_sgd(L):
        return __transformed_code_0_for_sgd(params, d_p_list, momentum_buffer_list, has_sparse_grad, foreach, weight_decay, momentum, lr, dampening, nesterov, maximize)
    # Note: this function might well not be executed directly. It might well be transformed again, i.e. adding one more guards and transformed code.
    return sgd(params, d_p_list, momentum_buffer_list, has_sparse_grad, foreach, weight_decay, momentum, lr, dampening, nesterov, maximize)

#============ end of sgd ============#
