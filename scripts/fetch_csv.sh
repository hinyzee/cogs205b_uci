#!/usr/bin/env bash
url="https://github.com/joachimvandekerckhove/cogs205b-s26/raw/refs/heads/main/modules/02-version-control/files/data.zip"
filename="data.zip"
wget -O "$filename" "$url"

temp_dir=$(mktemp -d)
unzip $filename -d $temp_dir

current_date=$(date +%F)
end_dir="data/$current_date"

mkdir -p "data"
mkdir -p "$end_dir"

# copy all csv files to end directory
shopt -s nullglob # does not throw error if no csv files are found
for file in "$temp_dir"/*.csv; do
    cp "$file" "$end_dir/"
done

# save and commit to git 
git add "$end_dir"
git commit -m "Adding CSV files for $current_date"
git push origin main


rm "$filename"
rm -rf "$temp_dir"
