#!/bin/bash
# Download Moatless vector indices for 93 instances.
# Usage: disconnect the VPN, then run bash download_indices.sh.

INDEX_STORE_URL="https://stmoatless.blob.core.windows.net/indexstore/20250118-voyage-code-3/"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INDEX_STORE_DIR="$SCRIPT_DIR/index_store"
INTERSECTION_FILE="$SCRIPT_DIR/../../swebench_lite_verified_intersection.json"

mkdir -p "$INDEX_STORE_DIR"

INSTANCE_IDS=$(python3 -c "
import json
with open('$INTERSECTION_FILE') as f:
    d = json.load(f)
print('\n'.join(d['instance_ids']))
")

TOTAL=$(echo "$INSTANCE_IDS" | wc -l | tr -d ' ')
COUNT=0
FAILED=()

echo "Start downloading $TOTAL indices to $INDEX_STORE_DIR"
echo "======================================================"

for INSTANCE_ID in $INSTANCE_IDS; do
    COUNT=$((COUNT + 1))
    INDEX_DIR="$INDEX_STORE_DIR/$INSTANCE_ID"

    if [ -d "$INDEX_DIR" ] && [ -n "$(ls -A "$INDEX_DIR" 2>/dev/null)" ]; then
        echo "[$COUNT/$TOTAL] $INSTANCE_ID - already exists, skip"
        continue
    fi

    echo "[$COUNT/$TOTAL] Downloading $INSTANCE_ID ..."
    TEMP=$(mktemp -d)
    ZIP="$TEMP/$INSTANCE_ID.zip"
    URL="${INDEX_STORE_URL}${INSTANCE_ID}.zip"
    CODE_FILE="$TEMP/code.txt"

    curl -L --max-time 120 --retry 3 --retry-delay 5 \
        -w '%{http_code}' -o "$ZIP" "$URL" > "$CODE_FILE" 2>/dev/null
    CURL_EXIT=$?
    HTTP_CODE=$(cat "$CODE_FILE")

    if [ $CURL_EXIT -ne 0 ]; then
        echo "  FAIL: curl error $CURL_EXIT (network/SSL issue)"
        FAILED+=("$INSTANCE_ID")
        rm -rf "$TEMP"
        continue
    fi

    if [ "$HTTP_CODE" != "200" ]; then
        echo "  FAIL: HTTP $HTTP_CODE"
        FAILED+=("$INSTANCE_ID")
        rm -rf "$TEMP"
        continue
    fi

    mkdir -p "$INDEX_DIR"
    if unzip -q "$ZIP" -d "$INDEX_DIR" 2>/dev/null; then
        # flatten one extra subdirectory if present
        FIRST=$(ls "$INDEX_DIR" 2>/dev/null | head -1)
        if [ -n "$FIRST" ] && [ -d "$INDEX_DIR/$FIRST" ] && [ "$(ls "$INDEX_DIR" | wc -l | tr -d ' ')" = "1" ]; then
            mv "$INDEX_DIR/$FIRST"/* "$INDEX_DIR/" 2>/dev/null || true
            rmdir "$INDEX_DIR/$FIRST" 2>/dev/null || true
        fi
        echo "  OK"
    else
        echo "  FAIL: unzip error"
        FAILED+=("$INSTANCE_ID")
        rm -rf "$INDEX_DIR"
    fi
    rm -rf "$TEMP"
done

echo ""
echo "======================================================"
echo "Done: $COUNT/$TOTAL, failed: ${#FAILED[@]}"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failed instances:"
    printf '  %s\n' "${FAILED[@]}"
fi
