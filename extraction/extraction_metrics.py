from typing import List

from extraction.event_schema import EventSchema
from extraction.predict_parser.predict_parser import Metric
from extraction.predict_parser.tree_predict_parser import TreePredictParser

decoding_format_dict = {
    'tree': TreePredictParser,
    'treespan': TreePredictParser,
}


def get_predict_parser(format_name):
    return decoding_format_dict[format_name]


def eval_pred(predict_parser, gold_list, pred_list, text_list=None, raw_list=None):
    
    well_formed_list, counter = predict_parser.decode(
        gold_list, pred_list, text_list, raw_list)
    


    event_metric = Metric()
    role_metric = Metric()
    TI_metric = Metric()
    TC_metric = Metric()
    AI_metric = Metric()
    AC_metric = Metric()
    strictTC_metric = Metric()
    strictAC_metric = Metric()

    for instance in well_formed_list:

        event_metric.count_instance(instance['gold_event'],
                                    instance['pred_event'])
        role_metric.count_instance(instance['gold_role'],
                                   instance['pred_role'],
                                   verbose=False)
        
        TI_metric.countTI(instance['gold_event'],  instance['pred_event'])
        TC_metric.countTC(instance['gold_event'],  instance['pred_event'])
        AC_metric.countAC(instance['gold_role'],  instance['pred_role'])
        AI_metric.countAI(instance['gold_role'],  instance['pred_role'])

        strictTC_metric.countStrictTC(instance['gold_event'],  instance['pred_event'])
        strictAC_metric.countStrictAC(instance['gold_role'],  instance['pred_role'])


    trigger_result = event_metric.compute_f1(prefix='trigger-')
    role_result = role_metric.compute_f1(prefix='role-')
    TI_result =TI_metric.compute_f1(prefix='TI-')
    TC_result =TC_metric.compute_f1(prefix='TC-')
    AI_result =AI_metric.compute_f1(prefix='AI-')
    AC_result =AC_metric.compute_f1(prefix='AC-')
    strictTC_result =strictTC_metric.compute_f1(prefix='strictTC-')
    strictAC_result =strictAC_metric.compute_f1(prefix='strictAC-')




    result = dict()
    result.update(trigger_result)
    result.update(role_result)
    result.update(TI_result)
    result.update(TC_result)
    result.update(AI_result)
    result.update(AC_result)
    result.update(strictTC_result)
    result.update(strictAC_result)
    result['AVG-F1'] = trigger_result.get('trigger-F1', 0.) + \
        role_result.get('role-F1', 0.)
    result.update(counter)
    return result


def get_extract_metrics(pred_lns: List[str], tgt_lns: List[str], label_constraint: EventSchema, decoding_format='tree'):
    predict_parser = get_predict_parser(format_name=decoding_format)(
        label_constraint=label_constraint)
    return eval_pred(
        predict_parser=predict_parser,
        gold_list=tgt_lns,
        pred_list=pred_lns
    )
