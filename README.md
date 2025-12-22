# dictycite

## dicty_curator_notes.py

Run a quick test on first 10
'''
python dicty_curator_notes.py --limit 10
'''

Run the full dataset
'''
python get_curated_notes.py --limit 0 --sleep-base 0.25 --sleep-jitter 0.10
'''

run with docker
'''
docker run -it -v "$PWD/output:/dictycite/output" --platform=linux/amd64 fulaibaowang/dictycite:22.12.2025 python dicty_curator_notes.py --limit 10
'''

