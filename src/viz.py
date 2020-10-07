import numpy as np
from imageio import imwrite
import os
import argparse
import json


def parse_arguments():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--color_cluster_path", type=str, default="./downloads/kmeans_centers.npy")
    parser.add_argument("--save_path", type=str, default="save")
    parser.add_argument("--load_path", type=str, default="Generated.Samples.From.iGPT.npy")
    
    args = parser.parse_args()
    print("input args:\n", json.dumps(vars(args), indent=4, separators=(",", ":")))
    return args
    
def main(args):
    
  import ipdb; ipdb.set_trace()  
  clusters = np.load(args.color_cluster_path)
  samples = np.load(args.load_path)    
  samples = [np.reshape(np.rint(127.5 * (clusters[s] + 1.0)), [32, 32, 3]).astype(np.uint8) for s in np.split(samples)]
  
  if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
  
  for i in range(samples.shape[0]):
    ind = curr_iter + i
    imwrite(f"{args.save_path}/sample_{ind}.png", samples[i])
    set_seed(args.seed)
    
    
if __name__ == "__main__":
    args = parse_arguments()
    main(args)
