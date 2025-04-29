#!/bin/bash

# Check if the source and target base directories are provided as arguments
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <source_directory> <target_base_directory>"
  exit 1
fi

SOURCE_DIR=$1
TARGET_BASE_DIR=$2

# Create separate directories for wave and force files
WAVE_TARGET_DIR="$TARGET_BASE_DIR/wave"
FORCE_TARGET_DIR="$TARGET_BASE_DIR/force"

# Check if the source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
  echo "Error: Source directory $SOURCE_DIR does not exist."
  exit 1
fi

# Function to clean directory name
clean_dirname() {
    # Remove numbers and special characters, convert to lowercase
    echo "$1" | tr -d '0-9' | tr -d '[:punct:]' | tr '[:upper:]' '[:lower:]' | tr -s ' ' | sed 's/^ *//g' | sed 's/ *$//g'
}

# Create temporary file to store wave directories
TEMP_FILE=$(mktemp)

# First, process wave files and store their base names
while IFS= read -r file; do
    # Get the grandparent directory name (one level up from audio folder)
    parent_dir=$(dirname "$(dirname "$file")")
    category_dir=$(basename "$parent_dir")
    base_name=$(basename "$parent_dir")
    
    # Clean the directory name
    clean_dir=$(clean_dirname "$category_dir")
    
    # Create the cleaned target directory for wave files
    new_target="$WAVE_TARGET_DIR/$clean_dir"
    mkdir -p "$new_target"
    
    # Copy the wave file
    cp "$file" "$new_target/"
    echo "Copied $(basename "$file") to $new_target/"
    
    # Store the base name in temporary file
    echo "$base_name" >> "$TEMP_FILE"
done < <(find "$SOURCE_DIR" -type f -name "*.wav")

# Then process force CSV files, but only if there's a matching wave file
while IFS= read -r file; do
    # Get the grandparent directory name (one level up from force folder)
    parent_dir=$(dirname "$(dirname "$file")")
    category_dir=$(basename "$parent_dir")
    base_name=$(basename "$parent_dir")
    
    # Check if we have a corresponding wave file
    if grep -q "^${base_name}$" "$TEMP_FILE"; then
        # Clean the directory name
        clean_dir=$(clean_dirname "$category_dir")
        
        # Create the cleaned target directory for force files
        new_target="$FORCE_TARGET_DIR/$clean_dir"
        mkdir -p "$new_target"
        
        # Copy the CSV file
        cp "$file" "$new_target/"
        echo "Copied $(basename "$file") to $new_target/"
    else
        echo "Skipping $(basename "$file") - no matching wave file found"
    fi
done < <(find "$SOURCE_DIR" -type f -name "*.csv")

# Clean up temporary file
rm "$TEMP_FILE"

echo "All files have been copied to their respective directories under $TARGET_BASE_DIR"
