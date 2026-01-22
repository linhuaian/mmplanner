for dir in /Users/SP12403/mmplanner/output/*/; do
    echo "Processing: $dir"
    python -u /Users/SP12403/mmplanner/eval/clip_metrics.py \
      --task_dir "$dir" \
      --device cpu \
      --local_files_only \
      --score_intermediate \
      --step_index_base 0 \
      --object_phrase_scoring \
      --num_object_phrases 5
done