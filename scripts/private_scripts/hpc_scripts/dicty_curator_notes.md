# running in frida hpc

make a tmp folder

```
mkdir tmp
cd tmp
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

inside the container:

```
cd /dictycite
python /dictycite/scripts/public/data_prep/dicty_curator_notes.py --limit 0 --sleep-base 0.35 --sleep-jitter 0.10 --out "/work/output/curator_notes.parquet" --claims-out "/work/output/curator_claims.parquet" --pubs-out   "/work/output/publications.parquet"
```
