#query="OPEN_ACCESS:y AND \"Dictyostelium discoideum\""
#query='TITLE:"Dictyostelium discoideum" AND TITLE:aggregation'
#pth=results/dicty_aggregation

#python fetch.py \
#    -q "$query" \
#    -xml \
#    -pdf \
#    --output_path "$pth"


query='TITLE_ABS:"Dictyostelium discoideum" AND TITLE_ABS:aggregation'

python fetch.py \
    --query "$query"

#python fetch.py --query '(dictyostelium) AND ( (HAS_FT:Y) AND (SRC:MED OR SRC:PMC ) )' --max_records 10 --get_text_from epmc --output_path dicty_papers