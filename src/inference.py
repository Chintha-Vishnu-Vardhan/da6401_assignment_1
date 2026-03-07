"""
Inference Script
Loads saved model and evaluates test dataset.
"""

import argparse
import json
import numpy as np

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path",type=str,default="src/best_model.npy")
    parser.add_argument("--config_path",type=str,default="src/best_config.json")

    return parser.parse_args()


def load_model(model_path,config_path):

    with open(config_path) as f:
        cfg=json.load(f)

    class Args:
        pass

    args=Args()

    for k,v in cfg.items():
        setattr(args,k,v)

    model=NeuralNetwork(args)

    weights=np.load(model_path,allow_pickle=True).item()

    model.set_weights(weights)

    return model,args


def evaluate(model,args):

    data=load_dataset(args.dataset)

    X_test,y_test_onehot,y_test_labels=data[4],data[5],data[6]

    logits=model.forward(X_test)

    preds=np.argmax(logits,axis=1)

    accuracy=accuracy_score(y_test_labels,preds)

    precision,recall,f1,_=precision_recall_fscore_support(
        y_test_labels,
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

    model,cfg=load_model(args.model_path,args.config_path)

    evaluate(model,cfg)


if __name__=="__main__":
    main()