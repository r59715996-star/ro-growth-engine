#!/bin/bash
set -euo pipefail

BATCH="${BATCH_PATH:-data/batch.json}"
INPUT=$(jq -r '.input' "$BATCH")
CLIP_COUNT=$(jq '.clips | length' "$BATCH")
OUTPUT_DIR="${CLIPS_DIR:-assets/clips}"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Temp directory for segments
tmpdir=$(mktemp -d)
trap "rm -rf $tmpdir" EXIT

# Crop filter for different views
crop_for_view() {
  case "$1" in
    guest)       echo "crop=(ih*9/16):ih:(iw-(ih*9/16))/2:0";;         # center
    split)       echo "crop=(ih*9/16):ih:((iw-(ih*9/16))*0.15):0";;   # pan 15% right
    split_right) echo "crop=(ih*9/16):ih:((iw-(ih*9/16))*0.85):0";;   # pan 15% left
    host)        echo "crop=(ih*9/16):ih:((iw-(ih*9/16))*0.85):0";;   # alias for split_right
    *)           echo "crop=(ih*9/16):ih:(iw-(ih*9/16))/2:0";;         # default center
  esac
}

# Process each clip
for ((c=0; c<$CLIP_COUNT; c++)); do
  START=$(jq -r ".clips[$c].start" "$BATCH")
  END=$(jq -r ".clips[$c].end" "$BATCH")
  OUTPUT="$OUTPUT_DIR/$(jq -r --argjson i "$c" '.clips[$i].output' "$BATCH")"
  
  # Check if clip has segments (for view switching)
  SEG_COUNT=$(jq --argjson i "$c" '.clips[$i].segments | length' "$BATCH")
  
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Processing clip $((c+1))/$CLIP_COUNT: $(basename "$OUTPUT")"
  echo "Time range: $START → $END"
  
  if [ "$SEG_COUNT" -eq 0 ]; then
    # No segments - simple straight cut with center crop
    echo "Mode: Simple cut (center crop)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    ffmpeg -y -loglevel error -stats \
      -ss "$START" -to "$END" \
      -i "$INPUT" \
      -vf "crop=(ih*9/16):ih:(iw-(ih*9/16))/2:0,scale=1080:1920" \
      -c:v libx264 -crf 18 -preset veryfast \
      -c:a aac \
      "$OUTPUT"
    
    echo "✅ Exported: $OUTPUT"
  
  else
    # Has segments - cut and stitch with view switching
    echo "Mode: View switching ($SEG_COUNT segments)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Convert HH:MM:SS to seconds for start time
    start_sec=$(echo "$START" | awk -F: '{print ($1*3600)+($2*60)+$3}')
    
    parts=()
    
    # Process each segment
    for ((i=0; i<$SEG_COUNT; i++)); do
      FROM=$(jq -r --argjson clip "$c" --argjson seg "$i" '.clips[$clip].segments[$seg].from' "$BATCH")
      TO=$(jq -r --argjson clip "$c" --argjson seg "$i" '.clips[$clip].segments[$seg].to' "$BATCH")
      VIEW=$(jq -r --argjson clip "$c" --argjson seg "$i" '.clips[$clip].segments[$seg].view' "$BATCH")
      
      # Calculate absolute timestamps
      abs_from=$(echo "$start_sec + $FROM" | bc)
      abs_to=$(echo "$start_sec + $TO" | bc)
      duration=$(echo "$abs_to - $abs_from" | bc)
      
      CROP=$(crop_for_view "$VIEW")
      part="$tmpdir/clip${c}_seg${i}.mp4"
      parts+=("$part")
      
      echo "  Segment $((i+1)): ${FROM}s → ${TO}s (${duration}s, view=$VIEW)"
      
      ffmpeg -y -loglevel error -stats \
        -ss "$abs_from" -t "$duration" \
        -i "$INPUT" \
        -vf "$CROP,scale=1080:1920" \
        -c:v libx264 -crf 18 -preset veryfast \
        -c:a aac \
        "$part"
    done
    
    # Create concat list
    listfile="$tmpdir/clip${c}_list.txt"
    for part in "${parts[@]}"; do
      echo "file '$part'" >> "$listfile"
    done
    
    # Concatenate segments
    echo "  Stitching segments..."
    ffmpeg -y -loglevel error -stats \
      -f concat -safe 0 \
      -i "$listfile" \
      -c copy \
      "$OUTPUT"
    
    echo "✅ Exported: $OUTPUT"
  fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 Done! Rendered $CLIP_COUNT clips in $OUTPUT_DIR/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
