"""
Inference Script
Evaluate trained model on test dataset.
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


def parse_arguments():

    parser=argparse.ArgumentParser()

    parser.add_argument("-d","--dataset",type=str,default="mnist",
                        choices=["mnist","fashion_mnist"])

    parser.add_argument("-e","--epochs",type=int,default=30)

    parser.add_argument("-b","--batch_size",type=int,default=128)

    parser.add_argument("-l","--loss",type=str,default="cross_entropy",
                        choices=["cross_entropy","mse"])

    parser.add_argument("-o","--optimizer",type=str,default="rmsprop",
                        choices=["sgd","momentum","nag","rmsprop"])

    parser.add_argument("-lr","--learning_rate",type=float,default=0.002)

    parser.add_argument("-wd","--weight_decay",type=float,default=0.0001)

    parser.add_argument("-nhl","--num_layers",type=int,default=3)

    parser.add_argument("-sz","--hidden_size",type=int,nargs="+",
                        default=[128,128,128])

    parser.add_argument("-a","--activation",type=str,default="tanh",
                        choices=["relu","sigmoid","tanh"])

    parser.add_argument("-w_i","--weight_init",type=str,default="xavier",
                        choices=["xavier","random"])

    parser.add_argument("-w_p","--wandb_project",type=str,default=None)

    parser.add_argument("--model_path",type=str,default="src/best_model.npy")

    parser.add_argument("--config_path",type=str,default="src/best_config.json")

    parser.add_argument("--val_split",type=float,default=0.1)

    parser.add_argument("--seed",type=int,default=42)

    return parser.parse_args()


def load_model(args):

    with open(args.config_path) as f:
        cfg=json.load(f)

    for k,v in cfg.items():
        setattr(args,k,v)

    model=NeuralNetwork(args)

    weights=np.load(args.model_path,allow_pickle=True).item()

    model.set_weights(weights)

    return model,args


def evaluate(model,args):

    data = load_dataset(args.dataset)

    X_test = data[4]
    y_test = data[6]

    logits=model.forward(X_test)

    preds=np.argmax(logits,axis=1)

    accuracy=accuracy_score(y_test,preds)

    precision,recall,f1,_=precision_recall_fscore_support(
        y_test,
        preds,
        average="macro",
        zero_division=0
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")


def main():

    args=parse_arguments()

    model,args=load_model(args)

    evaluate(model,args)


if __name__=="__main__":
    main()