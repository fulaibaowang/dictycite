# running in frida hpc

make a tmp folder

```
mkdir tmp1
cd tmp1
```

frida srun

```
srun -p dev --time=12:00:00 -c 4 \
  --container-image=fulaibaowang/dictycite:16.01.2026 \
  --container-mount-home \
  --container-mounts "${PWD}:/work${WORK:+,${WORK}:/work2}" \
  --container-workdir /work \
  --pty bash
```

inside the container, inscrease time out to 90 is necessary:

```
cd /dictycite
python /dictycite/scripts/public/dicty_publication.py --limit 0   --timeout 90 --sleep-base 0.35 --sleep-jitter 0.10 --out "/work/output/gene_publication_pmid.parquet"
```


## rerun with a list IDs
sometimes some IDS are not accessible at the moment like this:
```
DDB_G0288707: failed to fetch/parse gene page (500 Server Error: Internal Server Error for url: http://dictybase.org/gene/DDB_G0288707)
DDB_G3953806: failed to fetch/parse gene page (500 Server Error: Internal Server Error for url: http://dictybase.org/gene/DDB_G3953806)                                                                         
```

these genes can be rerun in a list with  --gene-list
```
python3 scripts/public/dicty_publication.py \
  --gene-list /work/failed_ids.txt \
  --out /work/output/gene_publication_pmid_rerun_failed.parquet \
  --timeout 60 --max-retries 4 \
  --sleep-base 0.5 --sleep-jitter 0.1
```

then merge two lists (overwrite those IDS with the results in retries).
