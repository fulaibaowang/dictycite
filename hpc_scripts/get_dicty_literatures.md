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

## warnings
Also with retry function, sometimes the server is temporary down, or the full text is really not accessible. I cannot avoid cases that fetching failed
```
Warning: failed to fetch text for PMC4863597: 404 Client Error: Not Found for url: https://www.ebi.ac.uk/europepmc/webservices/rest/PMC4863597/fullTextXML                                                                                                                                                                                                                                                                                                                        
```

construct a query with multiple PMCIDs manually
```
cat > pmcids_retry.txt << EOF
PMC11574339
PMC11443511
PMC10470824
PMC7554988
PMC7943716
PMC12618226
PMC10397335
PMC4863597
PMC5073093
PMC4940160
PMC4967854
PMC6080930
PMC6066244
PMC6262434
PMC6893374
PMC4792144
PMC4891362
PMC4931340
PMC3216324
PMC6593199
PMC4937987
PMC4730333
PMC3572115
PMC3927604
PMC4298681
PMC4436052
PMC3788181
PMC3321182
PMC5841363
PMC3083684
PMC4784174
EOF

# Convert to query string
QUERY=$(awk '{printf "%sPMCID:%s", (NR==1?"":" OR "), $1}' pmcids_retry.txt)
echo "$QUERY"

python fetch.py --query "$QUERY" --output_path /app/output/retry --get_text_from epmc_my 

```

If any IDs remain 404 error, let it be.
