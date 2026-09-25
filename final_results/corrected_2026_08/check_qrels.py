import collections, ir_datasets
names = ["beir/nfcorpus/test","beir/scifact/test","beir/arguana","beir/scidocs",
         "beir/fiqa/test","beir/trec-covid","beir/webis-touche2020/v2",
         "beir/quora/test","beir/nq","beir/dbpedia-entity/test",
         "beir/hotpotqa/test","beir/fever/test","beir/climate-fever"]
print(f"{'dataset':28s} {'relevance values':22s} {'binary':7s} {'metric-affected'}")
for n in names:
    try:
        ds = ir_datasets.load(n)
        c = collections.Counter(q.relevance for q in ds.qrels_iter())
        binary = set(c) <= {0,1}
        print(f"{n:28s} {str(dict(sorted(c.items()))):22s} {str(binary):7s} "
              f"{'no' if binary else 'YES'}", flush=True)
    except Exception as e:
        print(f"{n:28s} skip: {type(e).__name__}: {str(e)[:40]}", flush=True)
