import json, os

results = []
for bank in ['bank_A', 'bank_B', 'bank_C', 'bank_D']:
    with open(os.path.join('banks', bank, 'metadata.json')) as f:
        m = json.load(f)
    results.append(f"{bank}: train={m['n_train']}, val={m['n_val']}, test={m['n_test']}, fraud_rate={m['fraud_rate']:.4f}, features={m['feature_count']}")

with open('banks/bank_A/train.csv') as f:
    header = f.readline().strip().split(',')

with open('_inspect_out.txt', 'w') as f:
    for r in results:
        f.write(r + '\n')
    f.write(f"\nTotal CSV columns: {len(header)}\n")
    f.write(f"Target: {header[0]}\n")
    f.write(f"First 5: {header[:5]}\n")

print("Done - check _inspect_out.txt")
