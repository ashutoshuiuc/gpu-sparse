#!/bin/bash
# Evaluate Seismic retrieval quality (MRR@10, Recall@1000) on MS MARCO

# Resolve the repository root from this script's own location,
# so the script works from any checkout and any working directory.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -e

# Activate your own environment before running, or set CONDA_ENV to have this script do it.
if [ -n "${CONDA_ENV:-}" ]; then eval "$(conda shell.bash hook)"; conda activate "$CONDA_ENV"; fi
# Environment: the original run used a dedicated environment for this baseline.
if [ -n "${CONDA_ENV:-}" ]; then conda activate "$CONDA_ENV"; fi

GPU_SPARSE_DIR="${GPUSPARSE_ROOT:-$REPO_ROOT}"
DATA_DIR="${GPUSPARSE_DATA:-$REPO_ROOT/data}/seismic"
RESULTS_DIR="$GPU_SPARSE_DIR/final_results"

export CUDA_VISIBLE_DEVICES=""

python -c "
import os, sys, json, time, tarfile
import numpy as np
import seismic

data_dir = '$DATA_DIR'
results_dir = '$RESULTS_DIR'

# Seismic save() appends '.index.seismic' to path; load() needs the full filename
index_path = os.path.join(data_dir, 'seismic_index.bin.index.seismic')
if not os.path.exists(index_path):
    # Fallback: maybe it saved without the extension
    alt_path = os.path.join(data_dir, 'seismic_index.bin')
    if os.path.exists(alt_path):
        index_path = alt_path
    else:
        print(f'ERROR: Index not found at {index_path} or {alt_path}')
        print(f'Available files: {os.listdir(data_dir)}')
        sys.exit(1)
query_path = os.path.join(data_dir, 'queries.tar.gz')

# Load index
print('Loading Seismic index...')
index = seismic.SeismicIndex.load(index_path)
n_docs = index.len
print(f'Index loaded: {n_docs} documents')

# Load queries
print('Loading queries...')
string_type = seismic.get_seismic_string()
query_ids = []
query_components = []
query_values = []

with tarfile.open(query_path, 'r:gz') as tar:
    for member in tar.getmembers():
        f = tar.extractfile(member)
        if f is None:
            continue
        for line in f:
            line = line.decode('utf-8') if isinstance(line, bytes) else line
            line = line.strip()
            if not line:
                continue
            try:
                q = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = str(q.get('id', q.get('_id', q.get('qid', len(query_ids)))))
            vector = q.get('vector', {})
            if isinstance(vector, dict) and len(vector) > 0:
                tokens = list(vector.keys())
                values = [float(v) for v in vector.values()]
            else:
                continue
            query_ids.append(qid)
            query_components.append(np.array(tokens, dtype=string_type))
            query_values.append(np.array(values, dtype=np.float32))

n_queries = len(query_ids)
print(f'Loaded {n_queries} queries')

# Search all queries (k=1000 for recall evaluation)
print('Searching all queries (k=1000)...')
results = {}
for i in range(n_queries):
    if i % 1000 == 0:
        print(f'  {i}/{n_queries}...')
    hits = index.search(query_ids[i], query_components[i], query_values[i],
                        k=1000, query_cut=5, heap_factor=0.7)
    results[query_ids[i]] = hits
    if i == 0:
        print(f'  DEBUG: first query id={query_ids[0]}')
        print(f'  DEBUG: hits type={type(hits)}, len={len(hits)}')
        if len(hits) > 0:
            print(f'  DEBUG: first hit={hits[0]}, type={type(hits[0])}')
            if len(hits) > 1:
                print(f'  DEBUG: second hit={hits[1]}')

# Load qrels
print('Loading qrels...')
qrels_path = os.path.join(data_dir, 'qrels.dev.small.tsv')
if not os.path.exists(qrels_path):
    # Try downloading
    from huggingface_hub import hf_hub_download
    print('Downloading qrels...')
    # MS MARCO dev small qrels
    import urllib.request
    url = 'https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.small.tsv'
    urllib.request.urlretrieve(url, qrels_path)

qrels = {}
with open(qrels_path) as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 4:
            qid, _, did, rel = parts[0], parts[1], parts[2], int(parts[3])
        elif len(parts) == 3:
            qid, did, rel = parts[0], parts[1], int(parts[2])
        else:
            continue
        if rel > 0:
            if qid not in qrels:
                qrels[qid] = set()
            qrels[qid].add(did)

print(f'Loaded qrels for {len(qrels)} queries')
# Debug: show sample qrels
sample_qids = list(qrels.keys())[:3]
for sqid in sample_qids:
    print(f'  DEBUG qrels: qid={sqid}, relevant_docs={list(qrels[sqid])[:3]}')

# Compute MRR@10 and Recall@1000
mrr_at_10 = []
recall_at_10 = []
recall_at_100 = []
recall_at_1000 = []

evaluated = 0
for qid in query_ids:
    if qid not in qrels:
        continue
    relevant = qrels[qid]
    hits = results[qid]

    # Seismic search returns list of (query_id, score, document_id) tuples
    if isinstance(hits, list) and len(hits) > 0:
        if isinstance(hits[0], (list, tuple)) and len(hits[0]) >= 3:
            doc_ids = [str(h[2]) for h in hits]
        elif isinstance(hits[0], (list, tuple)) and len(hits[0]) == 2:
            doc_ids = [str(h[0]) for h in hits]
        else:
            doc_ids = [str(h) for h in hits]
    else:
        doc_ids = []

    # MRR@10
    rr = 0.0
    for rank, did in enumerate(doc_ids[:10]):
        if did in relevant:
            rr = 1.0 / (rank + 1)
            break
    mrr_at_10.append(rr)

    # Recall@k
    hits_set_10 = set(doc_ids[:10])
    hits_set_100 = set(doc_ids[:100])
    hits_set_1000 = set(doc_ids[:1000])

    recall_at_10.append(len(relevant & hits_set_10) / len(relevant))
    recall_at_100.append(len(relevant & hits_set_100) / len(relevant))
    recall_at_1000.append(len(relevant & hits_set_1000) / len(relevant))
    evaluated += 1

if evaluated > 0:
    mrr10 = np.mean(mrr_at_10)
    r10 = np.mean(recall_at_10)
    r100 = np.mean(recall_at_100)
    r1000 = np.mean(recall_at_1000)

    print(f'\nResults ({evaluated} queries evaluated):')
    print(f'  MRR@10: {mrr10:.4f}')
    print(f'  Recall@10: {r10:.4f}')
    print(f'  Recall@100: {r100:.4f}')
    print(f'  Recall@1000: {r1000:.4f}')

    output = {
        'system': 'Seismic',
        'dataset': 'msmarco-v1-splade',
        'n_docs': n_docs,
        'n_queries_evaluated': evaluated,
        'quality': {
            'mrr@10': float(mrr10),
            'recall@10': float(r10),
            'recall@100': float(r100),
            'recall@1000': float(r1000),
        },
        'config': {
            'n_postings': 3500,
            'summary_energy': 0.4,
            'centroid_fraction': 0.1,
            'query_cut': 5,
            'heap_factor': 10.0,
            'k': 1000,
        },
    }

    out_path = os.path.join(results_dir, 'seismic_quality_direct.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'Results saved to {out_path}')
else:
    print('ERROR: No queries could be evaluated (qrels mismatch)')
    sys.exit(1)
"

echo "=== Seismic quality evaluation complete ==="
