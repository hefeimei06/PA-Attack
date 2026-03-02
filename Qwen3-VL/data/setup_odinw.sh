#!/bin/bash

SOURCE_DIR="/home/Qwen3-VL/data/odinw_35"
TARGET_DIR="/home/Qwen3-VL/data/odinw"

mkdir -p "$TARGET_DIR"

DATASETS=(
    "AerialMaritimeDrone.zip|AerialMaritimeDrone"
    "Aquarium.zip|Aquarium"
    "CottontailRabbits.zip|Cottontail Rabbits"
    "EgoHands.zip|EgoHands"
    "NorthAmericaMushrooms.zip|NorthAmerica Mushrooms"
    "Packages.zip|Packages"
    "PascalVOC.zip|Pascal VOC"
    "pistols.zip|Pistols"
    "pothole.zip|Pothole"
    "Raccoon.zip|Raccoon"
    "ShellfishOpenImages.zip|ShellfishOpenImages"
    "thermalDogsAndPeople.zip|Thermal Dogs and People"
    "VehiclesOpenImages.zip|Vehicles OpenImages"
)

echo "🚀 Start unzip ODinW-13..."

for entry in "${DATASETS[@]}"; do
    IFS="|" read -r zip_name target_name <<< "$entry"
    
    ZIP_PATH="$SOURCE_DIR/$zip_name"
    
    if [ -f "$ZIP_PATH" ]; then
        echo "📦 Processing: $target_name ..."
        
        unzip -q -o "$ZIP_PATH" -d "$TARGET_DIR"
        
        raw_name="${zip_name%.zip}"
        
        if [ "$raw_name" != "$target_name" ]; then
            if [ -d "$TARGET_DIR/$raw_name" ]; then
                mv "$TARGET_DIR/$raw_name" "$TARGET_DIR/$target_name"
                echo "   -> Rename: $raw_name -> $target_name"
            elif [ -d "$TARGET_DIR/${raw_name^}" ]; then
                 mv "$TARGET_DIR/${raw_name^}" "$TARGET_DIR/$target_name"
                 echo "   -> Rename: ${raw_name^} -> $target_name"
            else
                echo "   ⚠️ Can not fine raw_name, please check $TARGET_DIR"
            fi
        fi
    else
        echo "❌ Can not find $ZIP_PATH"
    fi
done

echo "✅ All finished: $TARGET_DIR"