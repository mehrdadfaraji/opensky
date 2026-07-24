#!/bin/bash
# Stage 2: all 12 THETIS stroke classes (stage 1 was the 4 biomechanically
# distinct families: forehand_flat, backhand, flat_service, smash — kept the
# pipeline fast to build and validate). Now pulling the remaining 8
# within-family variants for the full vocabulary a real coach app needs.
#
# If this repo's already cloned from stage 1, this just adds the extra
# folders to the existing sparse-checkout — no need to re-clone.
#
# Run from inside month-05-tennis-coach/

set -e
mkdir -p data
cd data

if [ -d thetis/.git ]; then
    echo "Existing clone found — adding classes to it."
    cd thetis
else
    git clone --filter=blob:none --sparse https://github.com/THETIS-dataset/dataset.git thetis
    cd thetis
fi

git sparse-checkout set \
    VIDEO_RGB/forehand_flat VIDEO_RGB/backhand VIDEO_RGB/flat_service VIDEO_RGB/smash \
    VIDEO_RGB/forehand_openstands VIDEO_RGB/forehand_slice VIDEO_RGB/forehand_volley \
    VIDEO_RGB/backhand2hands VIDEO_RGB/backhand_slice VIDEO_RGB/backhand_volley \
    VIDEO_RGB/kick_service VIDEO_RGB/slice_service

echo "Done. Classes and clip counts:"
for d in VIDEO_RGB/*/; do
    echo "  $(basename "$d"): $(find "$d" -name '*.avi' | wc -l)"
done
echo "Total:"
find VIDEO_RGB -name "*.avi" | wc -l
