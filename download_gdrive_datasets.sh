#!/bin/bash

# Usage:
# ./download_gdrive.sh <destination_folder> <google_drive_id_or_url>

DEST="$1"
FILE_ID="$2"


mkdir -p "$DEST"

if ! command -v gdown &> /dev/null; then
    pip install gdown
fi

gdown "$FILE_ID" --folder -O "$DEST"
echo "Download completed"
