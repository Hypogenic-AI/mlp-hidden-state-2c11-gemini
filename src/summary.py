import numpy as np
import os
import glob

def summarize_cka():
    files = glob.glob('results/*_cka.npy')
    results = {}
    
    for f in files:
        name = os.path.basename(f).replace('_cka.npy', '')
        cka = np.load(f)
        adj = [cka[i, i+1] for i in range(len(cka)-1)]
        first_last = cka[0, -1]
        results[name] = {'mean_adj': np.mean(adj), 'first_last': first_last}
        
    print(f"{'Model Name':<50} | {'Mean Adj':<10} | {'First-Last':<10}")
    print("-" * 75)
    for name, res in sorted(results.items()):
        print(f"{name:<50} | {res['mean_adj']:<10.4f} | {res['first_last']:<10.4f}")

if __name__ == "__main__":
    summarize_cka()
