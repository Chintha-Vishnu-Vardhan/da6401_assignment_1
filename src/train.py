"""
Main Training Script
Train neural network and save best model + config for Gradescope.
"""

import argparse
import json
import os
import numpy as np

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset


def parse_arguments():

    parser = argparse.ArgumentParser(description="Train Neural Network")

    parser.add_argument("-d","--dataset",type=str,default="mnist",
                        choices=["mnist","fashion","fashion_mnist"])

    parser.add_argument("-e","--epochs",type=int,default=20)
    parser.add_argument("-b","--batch_size",type=int,default=128)

    parser.add_argument("-o","--optimizer",type=str,default="momentum",
                        choices=["sgd","momentum","nag","rmsprop"])

    parser.add_argument("-lr","--learning_rate",type=float,default=0.01)
    parser.add_argument("-wd","--weight_decay",type=float,default=0.0)

    parser.add_argument("-nhl","--num_layers",type=int,default=3)

    parser.add_argument("-sz","--hidden_size",
                        type=int,nargs="+",
                        default=[128,128,128])

    parser.add_argument("-a","--activation",type=str,default="relu",
                        choices=["relu","sigmoid","tanh"])

    parser.add_argument("-l","--loss",type=str,default="cross_entropy",
                        choices=["cross_entropy","mse"])

    parser.add_argument("-wi","--weight_init",type=str,default="xavier",
                        choices=["random","xavier","zeros"])

    parser.add_argument("--model_path",type=str,default="src/best_model.npy")
    parser.add_argument("--config_path",type=str,default="src/best_config.json")

    return parser.parse_args()


def save_config(args):

    config = {
        "dataset":args.dataset,
        "epochs":args.epochs,
        "batch_size":args.batch_size,
        "loss":args.loss,
        "optimizer":args.optimizer,
        "learning_rate":args.learning_rate,
        "weight_decay":args.weight_decay,
        "num_layers":args.num_layers,
        "hidden_size":args.hidden_size,
        "activation":args.activation,
        "weight_init":args.weight_init
    }

    with open(args.config_path,"w") as f:
        json.dump(config,f,indent=4)

    print("Saved config →",args.config_path)


def main():

    args = parse_arguments()

    data = load_dataset(args.dataset)

    X_train,y_train,X_val,y_val = data[0],data[1],data[2],data[3]

    model = NeuralNetwork(args)

    history = model.train(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        X_val=X_val,
        y_val=y_val
    )

    os.makedirs(os.path.dirname(args.model_path),exist_ok=True)

    np.save(args.model_path,model.get_weights())

    print("Saved model →",args.model_path)

    save_config(args)


if __name__ == "__main__":
    main()