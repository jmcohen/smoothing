# evaluate a smoothed classifier on a dataset
import argparse
import os
import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
import datetime
from architectures import get_architecture
from scipy.stats import norm
from torch.nn.functional import soft_margin_loss, cross_entropy
import math
import numpy as np
from math import ceil
import sys

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--radius", type=float, default=1.0)
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
args = parser.parse_args()


def compute_outer_loss(outputs: torch.tensor, target: int, p: float):
    ce = compute_inner_loss(outputs, target, 1.0)
    return torch.max(torch.zeros_like(ce, dtype=torch.float32).cuda(), 1 - p + ce)
    # return soft_margin_loss(p - ce, torch.ones_like(ce, dtype=torch.float32).cuda(), reduction='none') / math.log(2.0)


def compute_inner_loss(outputs: torch.tensor, target: int, t: float):
    batch_size = outputs.shape[0]
    ce = cross_entropy(t*outputs, torch.ones(batch_size, dtype=torch.long).cuda() * target, reduction='none') / math.log(2.0)
    return ce


def count_arr(arr: np.ndarray, length: int) -> np.ndarray:
    counts = np.zeros(length, dtype=int)
    for idx in arr:
        counts[idx] += 1
    return counts


def sample_noise(base_classifier, sigma, x: torch.tensor, y: int, num: int, batch_size, p: float) -> np.ndarray:
    """ Sample the base classifier's prediction under noisy corruptions of the input x.

    :param x: the input [channel x width x height]
    :param num: number of samples to collect
    :param batch_size:
    :return: an ndarray[int] of length num_classes containing the per-class counts
    """
    outer_losses = []
    inner_losses = []

    t_s = [0.1, 0.2, 0.4, 0.8, 1, 1.5, 2.0, 5.0]
    inner_losses_t = [[] for t in t_s]

    with torch.no_grad():
        counts = np.zeros(10, dtype=int)
        for _ in range(ceil(num / batch_size)):
            this_batch_size = min(batch_size, num)
            num -= this_batch_size

            batch = x.repeat((this_batch_size, 1, 1, 1))
            noise = torch.randn_like(batch, device='cuda') * sigma
            outputs = base_classifier(batch + noise)
            predictions = outputs.argmax(1)

            inner = compute_inner_loss(outputs, y, 1.0)
            outer = compute_outer_loss(outputs, y, p)

            outer_losses.extend(outer.cpu().numpy())
            inner_losses.extend(inner.cpu().numpy())

            for j in range(len(inner_losses_t)):
                inner_losses_t[j].append(compute_inner_loss(outputs, y, t_s[j]).cpu().numpy())

            counts += count_arr(predictions.cpu().numpy(), 10)

        inner_losses_t_means = [np.array(x).mean() for x in inner_losses_t]
        return counts, np.array(inner_losses).mean(), np.array(outer_losses).mean(), inner_losses_t_means


if __name__ == "__main__":
    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    base_classifier.load_state_dict(checkpoint['state_dict'])
    base_classifier.eval()

    dataset = get_dataset(args.dataset, args.split)


    bound3_total = 0
    count = 0


    # prepare output file
    f = sys.stdout if args.outfile == 'stdout' else open(args.outfile, 'w')
    print("p\tprob\terr\texp\tbnd 1\tbnd 2\tbnd 3\trun avg", file=f, flush=True)

    # iterate through the dataset
    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]

        p = 1.0 - norm.cdf(args.radius / args.sigma)

        x = x.cuda()
        counts, inner_loss, bound3, t_means = sample_noise(base_classifier, args.sigma, x, label, args.N, args.batch, p)

        prob = (args.N - counts[label]) / args.N
        true_loss = int(prob > p)

        bound1 = int(inner_loss > p)
        bound2 = soft_margin_loss(torch.tensor(p - inner_loss), torch.tensor(1.)).item() / math.log(2.0)

        print(t_means)

        count += 1
        bound3_total += bound3

        print("{:.3f}\t{:.3f}\t{}\t{:.3f}\t{}\t{:.3f}\t{:.3f}\t{:.3f}".format(p, prob, true_loss, inner_loss,
                                                                              bound1, bound2, bound3, bound3_total/count), file=f, flush=True)

    if f != sys.stdout:
        f.close()
