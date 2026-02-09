# running in frida hpc

make a tmp folder

```
mkdir tmp2
cd tmp2
```

frida srun

```
mkdir -p scripts/public/article_fetching/output
srun -p dev --time=12:00:00 -c 4 \
  --container-image=fulaibaowang/dictyfetch:19.01.2026 \
  --container-mount-home \
  --container-mounts "${PWD}/scripts/public/article_fetching/output:/app/output,${PWD}:/work" \
  --container-workdir /work \
  --pty bash
```

inside the container:

```
cd /dictycite/scripts/public/article_fetching

BASE_OUTPUT="/app/output"
mkdir -p "$BASE_OUTPUT"

# Loop through decades: 1980-2025
for start in 1980 1990 2000 2010 2020; do
  end=$((start + 9))
  output_dir="$BASE_OUTPUT/${start}-${end}"
  log_file="$BASE_OUTPUT/${start}-${end}.log"
  
  echo "=========================================="
  echo "Fetching articles from $start to $end..."
  echo "Output: $output_dir"
  echo "=========================================="
  
  python fetch.py \
    --query 'OPEN_ACCESS:y AND "Dictyostelium" AND FIRST_PDATE:[$start TO $end]' \
    --output_path "$output_dir" \
    --get_text_from epmc_my \
    2>&1 | tee "$log_file"
  
  echo "Batch $start-$end completed."
  echo ""
done

echo "All batches complete!"

# pre-1980
python fetch.py \
  --query 'OPEN_ACCESS:y AND "Dictyostelium" AND FIRST_PDATE:[1850 TO 1979]' \
  --output_path /app/output/1850-1979 \
  --get_text_from epmc_my \
    2>&1 | tee "$BASE_OUTPUT/1850-1979.log"

```

## warnings
Also with retry function, sometimes the server is temporary down, or the full text is really not accessible. I cannot avoid cases that fetching failed
```
Warning: failed to fetch text for PMC7049704: 503 Server Error: Service Temporarily Unavailable for url: https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7049704/fullTextXML             
```

construct a query with multiple PMCIDs manually in all.retry.txt
```
OUT=/app/output/retry
mkdir -p "$OUT"

i=0
while read -r pmc; do
  [ -z "$pmc" ] && continue
  i=$((i+1))
  echo "[$i] $pmc"

  # metadata + full text
  python fetch.py \
    --query "PMCID:$pmc" \
    --output_path "$OUT" \
    --get_text_from epmc_my

  # brief pause to be gentle on the API
  sleep 0.3
done < /app/output/all.retry.txt
```

If any IDs remain 404 error, let it be.
