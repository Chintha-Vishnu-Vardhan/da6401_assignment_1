"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import json
import os
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset


def parse_arguments():
    parser = argparse.ArgumentParser(description="Inference for Neural Network")

    parser.add_argument("-d","--dataset",type=str,default="mnist")

    parser.add_argument("-e","--epochs",type=int,default=10)
    parser.add_argument("-b","--batch_size",type=int,default=128)

    parser.add_argument("-o","--optimizer",type=str,default="momentum")
    parser.add_argument("-lr","--learning_rate",type=float,default=0.01)
    parser.add_argument("-wd","--weight_decay",type=float,default=0.0)

    parser.add_argument("-nhl","--num_layers",type=int,default=3)
    parser.add_argument("-sz","--hidden_size",type=str,nargs="+",
                        default=["128","128","128"])

    parser.add_argument("-a","--activation",type=str,default="relu")
    parser.add_argument("-l","--loss",type=str,default="cross_entropy")
    parser.add_argument("-wi","--weight_init",type=str,default="xavier")

    parser.add_argument("-w_p","--wandb_project",type=str,default=None)

    parser.add_argument("--model_path",type=str,default="src/best_model.npy")
    parser.add_argument("--config_save_path",type=str,default="src/best_config.json")

    parser.add_argument("--config_path",type=str,default=None)

    args = parser.parse_args()

    if args.config_path is not None:
        args.config_save_path = args.config_path

    return args


def normalize_hidden_sizes(args):

    hs = getattr(args,"hidden_size",[128,128,128])

    if isinstance(hs,str):
        hs = [int(x) for x in hs.replace("[","").replace("]","").split(",")]

    elif isinstance(hs,list):
        hs = [int(x) for x in hs]

    args.hidden_size = hs

    return args


def load_model_from_disk(model_path:str,config_path:str,args:Any)->NeuralNetwork:

    if os.path.exists(config_path):
        with open(config_path,"r",encoding="utf-8") as f:
            saved_config = json.load(f)

        for key,value in saved_config.items():
            setattr(args,key,value)

    args = normalize_hidden_sizes(args)

    weights = np.load(model_path,allow_pickle=True)

    if isinstance(weights,np.ndarray) and weights.ndim==0:
        weights = weights.item()

    model = NeuralNetwork(args)
    model.set_weights(weights)

    return model


def evaluate_model(model:NeuralNetwork,
                   X_test:np.ndarray,
                   y_test,
                   batch_size:int=512):

    n = X_test.shape[0]
    logits_list = []

    for start in range(0,n,batch_size):
        end = start+batch_size
        logits_list.append(model.forward(X_test[start:end]))

    logits = np.vstack(logits_list)

    y_pred = np.argmax(logits,axis=1)

    if y_test.ndim==2:
        y_true = np.argmax(y_test,axis=1)
    else:
        y_true = y_test

    accuracy = accuracy_score(y_true,y_pred)

    precision,recall,f1,_ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    try:
        model.last_logits = logits
        loss,_ = model.compute_loss_and_output(y_test)
    except:
        loss = 0.0

    return {
        "logits":logits,
        "loss":float(loss),
        "accuracy":float(accuracy),
        "precision":float(precision),
        "recall":float(recall),
        "f1":float(f1)
    }


def main():

    args = parse_arguments()

    model = load_model_from_disk(
        args.model_path,
        args.config_save_path,
        args
    )

    data = load_dataset(args.dataset)

    X_test = data[4]
    y_test = data[5]

    results = evaluate_model(
        model,
        X_test,
        y_test,
        batch_size=args.batch_size
    )

    print(
        f"Loss: {results['loss']:.4f}\n"
        f"Accuracy: {results['accuracy']:.4f}\n"
        f"Precision: {results['precision']:.4f}\n"
        f"Recall: {results['recall']:.4f}\n"
        f"F1-score: {results['f1']:.4f}"
    )

    return results


if __name__ == "__main__":
    main()