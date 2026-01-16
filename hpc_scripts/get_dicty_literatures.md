# running in frida hpc

make a tmp folder

```
mkdir tmp2
cd tmp2
```

frida srun

```
mkdir -p article_fetching/output
srun -p dev --time=12:00:00 -c 4 \
  --container-image=fulaibaowang/dictyfetch:14.01.2026 \
  --container-mount-home \
  --container-mounts "${PWD}/article_fetching/output:/app/output,${PWD}:/work" \
  --container-workdir /work \
  --pty bash
```

inside the container:

```
cd /dictycite/article_fetching


BASE_OUTPUT="/app/output"
mkdir -p "$BASE_OUTPUT"

# Loop through decades: 1980-2025
for start in 1980 1990 2000 2010 2020; do
  end=$((start + 9))
  output_dir="$BASE_OUTPUT/${start}-${end}"
  
  echo "=========================================="
  echo "Fetching articles from $start to $end..."
  echo "Output: $output_dir"
  echo "=========================================="
  
  python fetch.py \
    --query "OPEN_ACCESS:y AND \"Dictyostelium discoideum\" AND FIRST_PDATE:[$start TO $end]" \
    --output_path "$output_dir" \
    --get_text_from epmc_my
    --delay 0.2
  
  echo "Batch $start-$end completed."
  echo ""
done

echo "All batches complete!"

# pre-1980
python fetch.py \
  --query 'OPEN_ACCESS:y AND "Dictyostelium discoideum" AND FIRST_PDATE:[1850 TO 1979]' \
  --output_path /app/output/1850-1979 \
  --get_text_from epmc_my

```
