# import ipdb
# import numpy as np
# from typing import List
# from evaluate import load
# from tqdm import tqdm

# bertscore = load('bertscore', model_type='xlm-roberta-base')

# def compute_all_pairs_bertscore(preds: List[str], refs: List[str]):
#     """Compute the bertscore between all pairs of predictions and references.

#     Args:
#         preds (List[str]): A list of predictions
#         refs (List[str]): A list of references

#     Returns:
#         bertscore (np.array): A list of bertscore values for each pair of predictions and references [len(preds) x len(refs)]

#     """
#     refs_recalls = []
#     for ref in tqdm(refs):
#         metric_for_ref = bertscore.compute(predictions=preds, references=([ref] * len(preds)), model_type='xlm-roberta-base') # should be a list of length len(preds)
#         refs_recalls.append(metric_for_ref['f1'])
#     return np.array(refs_recalls).T
    
#     # results = bertscore.compute(predictions=preds, references=[refs], model_type='xlm-roberta-base')

#     # ipdb.set_trace()