"""
Main Training Script
Entry point for training NumPy MLP for MNIST/Fashion-MNIST.
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset


def parse_arguments():

    parser = argparse.ArgumentParser()

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


def main():

    args=parse_arguments()

    np.random.seed(args.seed)

    data = load_dataset(args.dataset)

    X_train = data[0]
    y_train = data[1]
    X_val   = data[2]
    y_val   = data[3]

    model = NeuralNetwork(args)

    print("\nTraining model...")

    model.train(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    os.makedirs(os.path.dirname(args.model_path),exist_ok=True)

    weights=model.get_weights()

    np.save(args.model_path,weights)

    print("Saved model →",args.model_path)

    config={
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
        json.dump(config,f,indent=2)

    print("Saved config →",args.config_path)

    print("\nTraining complete")


if __name__=="__main__":
    main()