# A tool for synthesizing training data by combining WAV audio, CLAP embeddings, and VAE outputs for the Amateur-Professional Semantic Divide research.

## Project Structure
```
├── scripts
│   ├── backup-csv.sh                    # Backup CSV files
│   ├── bandcamp_filename_mapper.py      # Build URL encoding for bandcamp dataset
│   ├── clap_merge.py                    # Merge CLAP files
│   ├── convert_dyqy.py                  # Convert unopenable dyqy files
│   ├── decompress.py                    # Decompress MP3 files
│   ├── make_dataset_csv.py             # Convert merged dataset and songid CSV format to separate format
│   ├── vae_vec_merge.py                # Merge VAE files
│   └── wave_merge.py                   # Merge WAV files
└── test
    └── test_data_merge.py              # Test WAV merge results
```

This tool is designed to support the MuChin dataset processing pipeline for targeted training experiments.