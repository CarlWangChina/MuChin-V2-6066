src_base="/nfs/data/clap/merged"
dst_base="/nfs/data/clap/merged-backup"

find "$src_base" -type f -name "clap_metas.csv" | while read src_file; do
  relative_path="${src_file
  dst_file="$dst_base$relative_path"

  dst_dir=$(dirname "$dst_file")
  mkdir -p "$dst_dir"

  cp "$src_file" "$dst_file"

  echo "Copied $src_file to $dst_file"
done