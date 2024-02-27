import json
import os
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from collections import namedtuple
# from utils.eval_utils import compute_f1

def compute_f1(predicts, labels):
    tp = 0
    pos = 0
    true = 0
    for p, l in zip(predicts, labels):
        true += len(l)
        pos += len(p)
        tp += len(p & l)
    precision = tp / pos if pos != 0 else 0
    recall = tp / true if true != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
    return precision, recall, f1

class Task(object):

    def __init__(self, task_name, task_elements) -> None:
        self.task_name = task_name
        self.task_elements = task_elements
        self.Result = namedtuple(task_name, task_elements)
        self.contexts = []
        self.predicts = []
        self.labels = []

    def _json_to_result(self, jobj):
        e = ["NULL" if jobj[k] is None else jobj[k] for k in self.task_elements]
        return self.Result(*e)

    def parse(self, context, predict, label):
        p_set = set(self._json_to_result(each) for each in predict)
        l_set = set(self._json_to_result(each) for each in label)
        if "Implicity" in self.task_name:
            p_set = set(each for each in p_set if each.target == "NULL")
            l_set = set(each for each in l_set if each.target == "NULL")
            # if "Implicity_Only" in self.task_name:
            #     if all(each.target == "NULL" for each in l_set):
            #         self.contexts.append(context)
            #         self.predicts.append(p_set)
            #         self.labels.append(l_set)
            # else:
            #     if any(each.target == "NULL" for each in l_set):
            #         self.contexts.append(context)
            #         self.predicts.append(p_set)
            #         self.labels.append(l_set)
        # else:
        elif "Mixed" in self.task_name:
            if len(set((each.aspect, each.polarity) for each in l_set)) <= 1: 
                return
                
        self.contexts.append(context)
        self.predicts.append(p_set)
        self.labels.append(l_set)

    def evaluate(self):
        return compute_f1(self.predicts, self.labels)

    def reset(self):
        self.contexts = []
        self.predicts = []
        self.labels = []


class MultiTaskEvaluatior(object):

    def __init__(self) -> None:
        task_config = {
            "A": ["aspect"],
            "T": ["target"],
            "TA": ["target", "aspect"],
            "AS": ["aspect", "polarity"],
            "TS": ["target", "polarity"],
            "TAS": ["target", "aspect", "polarity"],
            "TAS_Implicity_Only": ["target", "aspect", "polarity"],
            "Mixed_sentence": ["target", "aspect", "polarity"],
        }
        self.task_pools = {}
        for name, elements in task_config.items():
            self.task_pools[name] = Task(name, elements)

        self.eval_results = {k: {"p": [], "r": [], "f1": []}
                             for k in task_config}

    def eval_epoch_task(self, epoch, task_name):
        self.parse(epoch)
        self.eval(task_name)

    def parse(self, contexts, preds, labels):
        for c, p, l in zip(contexts, preds, labels):
            json_p = [{'target': item[0], 'aspect': item[1], 'polarity': item[2]} for item in p]
            json_l = [{'target': item[0], 'aspect': item[1], 'polarity': item[2]} for item in l]
            for task in self.task_pools.values():
                task.parse(c, json_p, json_l)

    def eval(self, task_name):
        task = self.task_pools[task_name]
        p, r, f1 = task.evaluate()
        self.eval_results[task.task_name]["p"].append(p)
        self.eval_results[task.task_name]["r"].append(r)
        self.eval_results[task.task_name]["f1"].append(f1)

    def reset(self):
        for task in self.task_pools.values():
            task.reset()

    def display(self, save=False):
        best_results = []
        for name, results in self.eval_results.items():
            best_epoch = np.argmax(results["f1"])
            best_results.append({
                "epoch": best_epoch+1,
                "p": results["p"][best_epoch],
                "r": results["r"][best_epoch],
                "f1": results["f1"][best_epoch]
            })
        df = pd.DataFrame(data=best_results, index=self.eval_results.keys())
        if save:
            df.to_csv(os.path.join(self.output_dir, "eval_results.csv"))

        print(df)
        print()
        return df

    def full_evaluate(self, contexts, preds, labels, save=False):
        self.parse(contexts, preds, labels)
        for task_name in self.task_pools:
            self.eval(task_name)
        return self.display(save=save)


if __name__ == "__main__":
    contexts = [['Love', 'Al', 'Di', 'La'], ['I', 'recommend', 'this', 'place', 'to', 'everyone', '.'], ['One', 'of', 'my', 'favorite', 'places', 'in', 'Brooklyn', '.']]
    all_preds = [[('Al Di La', 'restaurant general', 'positive')], [('place', 'restaurant general', 'positive')], [('NULL', 'restaurant general', 'positive')]]
    all_labels = [[('Al Di La', 'restaurant general', 'positive')], [('place', 'restaurant general', 'positive')], [('NULL', 'restaurant general', 'positive')]]
    evaluator = MultiTaskEvaluatior()
    evaluator.full_evaluate(contexts, all_preds, all_labels, False)




        
