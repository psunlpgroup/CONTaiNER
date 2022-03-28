
import numpy as np
import glob
import os
import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir",
            default=None,
            type=str,
            required=True,
            help="The input data dir.",)
    parser.add_argument("--range",
                        default=5000,
                        type=int,
                        required=False,
                        help="ranging the amount of data to consider in count")
    args = parser.parse_args()
    path = args.target_dir
    s = []

    target_res_text = 'results.txt'
    if args.range == None:
        s = glob.glob(os.path.join(path, '*', target_res_text))
    else:
        for i in range(0, args.range):
            try:
                s.append(glob.glob(os.path.join(path, str(i), target_res_text))[0])
            except:
                print("Missing file: " + str(i))

    precisions = []
    recalls = []
    f1s = []
    pred_sum = []
    tp_sum = []
    true_sum = []

    for file in s:
        with open(file) as f:
            lines = f.readlines()
            pred_sum.append(int(lines[6].split()[-1]))
            tp_sum.append(int(lines[8].split()[-1]))
            true_sum.append(int(lines[9].split()[-1]))
    recall = np.sum(tp_sum) / np.sum(true_sum)
    precision = np.sum(tp_sum) / np.sum(pred_sum)
    f1 = (2 * precision * recall) / (precision + recall)

    print("avg. f1 = %f" % (f1) )
    print("avg. precision = %f" % (precision))
    print("avg. recall = %f" % (recall))
    print("covered = %f" % len(tp_sum))




if __name__ == "__main__":
    main()