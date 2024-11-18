import torch
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from .attacks.pgd_idscore import PGD
import numpy as np
import gc
from .id_scores.msp import get_msp

def mean_id_score_diff(model,
                       dataloader,
                       device=None,
                       verbose=False,
                       eps=1/255):

    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    
    attack_eps = eps
    attack_steps = 10
    attack_alpha = 2.5 * attack_eps / attack_steps
    
    attack = PGD(model, eps=attack_eps, steps=attack_steps, alpha=attack_alpha)
    before_attack_scores = []
    after_attack_scores = []
    
    for data, targets in dataloader:
        data = data.to(device)
        
        before_attack = get_msp(model, data)
        
        data = attack(data, targets)
        
        after_attack = get_msp(model, data)
        
        before_attack_scores += before_attack.detach().cpu().numpy().tolist()
        after_attack_scores += after_attack.detach().cpu().numpy().tolist()
        
        torch.cuda.empty_cache()
        gc.collect()
        
    before_attack_scores = np.asarray(before_attack_scores)
    after_attack_scores = np.asarray(after_attack_scores)
    
    if verbose:
        print("Before:", np.mean(before_attack_scores))
        print("After:", np.mean(after_attack_scores))
    
    return np.mean(after_attack_scores) - np.mean(before_attack_scores)

def get_models_scores(model_dataset,
                        model_score_function,
                        progress,
                        verbose=False,
                        live=True,
                        strict=False):
    
    labels = []
    scores = []

    tq = range(len(model_dataset))
    if progress is False:
        tq = tqdm(tq)
    
    if live:
        seen_labels = set()
    failed_models = 0
    
    for i in tq:
        try:
            model, label = model_dataset[i]

            score = model_score_function(model)
            if progress:
                print(f'No. {i}, Label: {label}, Score: {score}')
            
            scores.append(score)
            labels.append(label)
            if live:
                if verbose:
                    print("Label:", label, "Score:", score)
                seen_labels.add(label)
                
                if len(seen_labels) > 1:
                    print("Current:", roc_auc_score(labels, scores))
        except Exception as e:
            if strict:
                raise e
            failed_models += 1
            print(f"The following error occured during the evaluation of a model: {str(e)}")
            print("Skipping this model")
    print("No. of failed models:", failed_models)
    return scores, labels

def get_results(scores, labels):
    return roc_auc_score(labels, scores)

def evaluate_modelset(model_dataset,
                        signature_function,
                        signature_function_kwargs={},
                        get_dataloader_func=None,
                        verbose=False,
                        strict=False,
                        progress=False):
    
    def model_score_function(model):
        dataloader = get_dataloader_func()
        
        return signature_function(model,
                                 dataloader,
                                 **signature_function_kwargs)
    
    scores, labels = get_models_scores(model_dataset,
                                       model_score_function,
                                       progress,
                                       strict=strict,
                                       verbose=verbose)
    
    return get_results(scores, labels)
