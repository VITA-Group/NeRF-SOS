import os, sys
import numpy as np

if __name__ == "__main__":
    # Convert n * (o - p) = 0 to n * p = D
    if len(sys.argv) != 3:
        print("Usage: %s <input_path> <output_path>" % (sys.argv[0], sys.argv[1]))
    
    from_arr = np.load(sys.argv[1])
    print("Loading from %s => (%d, %d)" % (sys.argv[1], from_arr.shape[0], from_arr.shape[1]))
    
    norms = from_arr[:, :3]
    points = from_arr[:, 3:]
    D = np.sum(norms * points, axis=-1, keepdims=True)
    to_arr = np.concatenate([norms, D], axis=-1)
    print("Saving to %s => (%d, %d)" % (sys.argv[1], to_arr.shape[0], to_arr.shape[1]))
    np.save(sys.argv[2], to_arr)