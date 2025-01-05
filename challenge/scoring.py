# The scoring program compute scores from:
# - The ground truth
# - The predictions made by the candidate model

# Imports
import json
import os
import sys


# Path
input_dir = ''    # Input from ingestion program
output_dir = '' # To write the scores
# input_dir = sys.argv[1]       # Input from ingestion program
# output_dir = sys.argv[2]      # To write the scores
reference_dir = os.path.join(input_dir, 'ref')  # Ground truth data
prediction_dir = os.path.join(input_dir, 'res') # Prediction made by the model
score_file = os.path.join(output_dir, 'scores.json')          # Scores
html_file = os.path.join(output_dir, 'detailed_results.html') # Detailed feedback

def write_file(file, content):
    """ Write content in file.
    """
    with open(file, 'a', encoding="utf-8") as f:
        f.write(content)

def get_dataset_names():
    """ Return the names of the datasets.
    """
    return ['dataset1', 'dataset2', 'dataset3', 'dataset4']


def print_bar():
    """ Display a bar ('----------')
    """
    print('-' * 10)



class Metric:
    def __init__(self):
        self.tp = 0.
        self.gold_num = 0.
        self.pred_num = 0.

    @staticmethod
    def safe_div(a, b):
        if b == 0.:
            return 0.
        else:
            return a / b

    def compute_f1(self, prefix=''):
        tp = self.tp
        pred_num = self.pred_num
        gold_num = self.gold_num
        p, r = self.safe_div(tp, pred_num), self.safe_div(tp, gold_num)
        return {prefix + 'tp': tp,
                prefix + 'gold': gold_num,
                prefix + 'pred': pred_num,
                prefix + 'P': p * 100,
                prefix + 'R': r * 100,
                prefix + 'F1': self.safe_div(2 * p * r, p + r) * 100
                }





    def countStrictTC(self, gold_list, pred_list, verbose=False): #   'gold_event': [('Nominate', 'named')], 'pred_event': [('Start-Position', 'named')],

        if verbose:
            print("Gold:", gold_list)
            print("Pred:", pred_list)


        self.gold_num += len(gold_list)
        self.pred_num += len(pred_list)

        goldTC = []
        for goldEvent in gold_list:
            goldTC.append((goldEvent[0],goldEvent[1].lower()))



        for pred in pred_list:
            if (pred[0],pred[1].lower()) in goldTC:
                self.tp += 1
                goldTC.remove((pred[0],pred[1].lower()))


    
    def countStrictAC(self, gold_list, pred_list, verbose=False):

        if verbose:
            print("Gold:", gold_list)
            print("Pred:", pred_list)


        self.gold_num += len(gold_list)
        self.pred_num += len(pred_list)

        goldAC = []
        for goldArg in gold_list:
            goldAC.append((goldArg[0],goldArg[1],goldArg[2].lower()))



        for pred in pred_list:
            if (pred[0],pred[1],pred[2].lower()) in goldAC:
                self.tp += 1
                goldAC.remove((pred[0],pred[1],pred[2].lower()))



    def countRecord(self, gold_list, pred_list, verbose=False):

        if verbose:
            print("Gold:", gold_list)
            print("Pred:", pred_list)


        self.gold_num += len(gold_list)
        self.pred_num += len(pred_list)

        goldRecord = []
        for goldArg in gold_list:
            goldRecord.append((goldArg[0],goldArg[1].lower(),goldArg[2],goldArg[3].lower()))



        for pred in pred_list:
            if (pred[0],pred[1].lower(),pred[2],pred[3].lower()) in goldRecord:
                self.tp += 1
                goldRecord.remove((pred[0],pred[1].lower(),pred[2],pred[3].lower()))




def extract_ED(json_data):
    result = []
    for instance in json_data:
        res_list = []
       
        event = instance['event']
        if event:
            for one_item in event:
                res_list.append((one_item['type'],one_item['trigger']))
            
        result.append(res_list)
        
    return result


def extract_EAE(json_data):
    result = []
    for instance in json_data:
        res_list = []

        event = instance['event']

        if event:
            for one_item in event:
                if one_item['arguments']:
                    for one_argu in one_item['arguments']:
                        res_list.append((one_item['type'], one_argu['role'],one_argu['name']))

        result.append(res_list)

    return result


def extract_record(json_data):
    result = []
    for instance in json_data:
        res_list = []

        event = instance['event']

        if event:
            for one_item in event:
                if one_item['arguments']:
                    for one_argu in one_item['arguments']:
                        res_list.append((one_item['type'],one_item['trigger'], one_argu['role'],one_argu['name']))

        result.append(res_list)

    return result



        

def main():
    """ The scoring program.
    """
    print_bar()
    print('Scoring program.')
    # Initialized detailed results
    # write_file(html_file, '<h1>Detailed results</h1>') # Create the file to give real-time feedback
    scores = {}
    

    try:
        reference_path = os.path.join(reference_dir, 'gold.json')
        reference = json.load(open(reference_path))

        prediction_path = os.path.join(prediction_dir, 'results.json')
        prediction = json.load(open(prediction_path))
        
        task1_metric = Metric()
        task2_metric = Metric()
        task3_metric = Metric()
        
        # task1
        pred_task1 = extract_ED(prediction)
        gold_task1 = extract_ED(reference)
        
        for idx,instance in enumerate(pred_task1):
            task1_metric.countStrictTC(gold_task1[idx],pred_task1[idx],verbose=False)
            
        task1_score = task1_metric.compute_f1(prefix='task1-')
        print(task1_score)

        # task2
        pred_task2 = extract_EAE(prediction)
        gold_task2 = extract_EAE(reference)

        for idx, instance in enumerate(pred_task2):
            task2_metric.countStrictAC(gold_task2[idx], pred_task2[idx], verbose=False)

        task2_score = task2_metric.compute_f1(prefix='task2-')
        print(task2_score)


        # task3
        pred_task3 = extract_record(prediction)
        gold_task3 = extract_record(reference)

        for idx, instance in enumerate(pred_task3):
            task3_metric.countStrictAC(gold_task3[idx], pred_task3[idx], verbose=False)

        task3_score = task3_metric.compute_f1(prefix='task3-')
        print(task3_score)
        scores['task1_score'] = task1_score['task1-F1']
        scores['task2_score'] = task2_score['task2-F1']
        scores['task3_score'] = task3_score['task3-F1']

        overall_score = 0.3 * task1_score['task1-F1'] + 0.3 * task2_score['task2-F1'] + 0.2 * task3_score['task3-F1']
        print(f'overall_score: {overall_score}')
        scores['overall_score'] = overall_score




        

    except Exception as e:
        print(e)
   

    print_bar()
    print('Scoring program finished. Writing scores.')

    write_file(score_file, json.dumps(scores))


if __name__ == '__main__':
    main()
