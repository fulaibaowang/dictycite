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
python /dictycite/dicty_curator_notes.py --limit 0   --timeout 90 --sleep-base 0.5 --sleep-jitter 0.10 --out "/work/output/gene_publication_pmid.parquet"
```


