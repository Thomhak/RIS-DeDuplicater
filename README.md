# RIS File Deduplicator

A Python tool for identifying and removing duplicate bibliographic records from RIS files. Designed for researchers managing literature searches from multiple databases, particularly useful for systematic reviews and meta-analyses.

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🔍 **Duplicate detection** using DOI matching and text similarity analysis
- ⚙️ **Configurable thresholds** with similarity settings from 0.0 to 1.0
- 🔗 **Multi-database support** (Scopus, PubMed, Web of Science, IEEE, CAB Abstracts, etc.)
- 🔄 **Information merging** preserves complete bibliographic data
- 🛡️ **File parsing** handles encoding issues and malformed files

## Installation

**Requirements:** Python 3.6+ (no additional dependencies)

```bash
# Clone repository
git clone https://github.com/yourusername/ris-deduplicator.git
cd ris-deduplicator

# Or download directly
curl -O https://raw.githubusercontent.com/yourusername/ris-deduplicator/main/ris_deduplicator.py
```

## Quick Start

```bash
# Basic usage with default settings
python3 ris_deduplicator.py *.ris

# Systematic review with detailed logging
python3 ris_deduplicator.py database_*.ris --show-duplicates -v -o systematic_review.ris

# Adjust threshold (0.75=relaxed, 0.80=balanced, 0.95=conservative)
python3 ris_deduplicator.py *.ris --threshold 0.75
```

## Usage

```
python3 ris_deduplicator.py [files] [-o output.ris] [--threshold 0.80] [--show-duplicates] [--verbose]
```

**Key Arguments:**
- `files`: RIS files to process (supports wildcards)
- `--threshold`: Similarity threshold (0.0-1.0, default: 0.80)
- `--show-duplicates`: Show detailed duplicate analysis
- `--verbose`: Enable detailed progress reporting

**Threshold Guidelines:**
- **0.95-1.0**: Conservative (high precision, may miss some duplicates)
- **0.85-0.94**: Balanced approach (good for high-quality databases)
- **0.75-0.84**: Recommended (good precision/recall balance)
- **<0.75**: Relaxed (finds more duplicates, review for false positives)

## How It Works

1. **DOI Matching**: Identical DOIs automatically identify duplicates (highest confidence)
2. **Similarity Analysis**: Multi-factor scoring for non-DOI records:
   - **Title matching**: Combines multiple similarity algorithms (Jaccard, Levenshtein, sequence matching)
   - **Author comparison**: Handles name variations, initials, and different formats
   - **Text normalization**: Unicode handling, abbreviation expansion, stopword removal
   - **Journal name matching**: Abbreviation database and pattern matching
   - **Publication year tolerance**: ±1 year flexibility for data entry variations
3. **Scoring**: Weighted scoring based on available data quality
4. **Merging**: Combines information while preserving all bibliographic data
5. **Output**: Standard RIS format compatible with reference managers

## Example Output

```
🔧 Running RIS Deduplicator v1.4.0
🔧 Similarity threshold: 0.80 (balanced)
📂 Processing 3 RIS file(s)...
📊 Total records read: 2,306
🔍 Finding duplicates...
Found 599 duplicate groups with 945 duplicate records
  • 566 groups matched by DOI
  • 33 groups matched by similarity
✅ Successfully wrote 1,361 deduplicated records
🧹 Removed 945 duplicates (41.0% reduction)
```

**Key Features:**
- Similarity-based duplicate detection for records without DOIs
- Handles database formatting variations  
- Author name and journal matching across different databases

## Use Cases

**Academic Literature Reviews:**
```bash
# Multi-database deduplication
python3 ris_deduplicator.py scopus.ris pubmed.ris ieee.ris -o literature_review.ris
```

**Systematic Reviews:**
```bash
# With documentation for methodology
python3 ris_deduplicator.py database_*.ris --threshold 0.80 -v --show-duplicates -o systematic_review.ris
```

**Cross-Database Integration:**
```bash
# Multiple databases with different formatting
python3 ris_deduplicator.py cab_abstracts.ris scopus.ris wos.ris --threshold 0.75 -o integrated_results.ris
```

**High-Quality Databases:**
```bash
# Conservative settings for clean, well-formatted data
python3 ris_deduplicator.py *.ris --threshold 0.90 -o clean_refs.ris
```

## What's New in v1.4.0

- **Improved similarity algorithms**: Better non-DOI duplicate detection
- **Text processing**: Unicode normalization and abbreviation expansion
- **Author matching**: Handles name variations, initials, and different formats
- **Journal normalization**: Abbreviation database for matching
- **Updated thresholds**: New default of 0.80 for better precision/recall balance
- **Cross-database support**: Better handling of formatting differences between databases

## Requirements

- Python 3.6 or higher
- No external dependencies required
- Compatible with all major operating systems (Windows, macOS, Linux)

## Performance

Tested on datasets with 2,000+ records from multiple databases:
- **Processing speed**: ~1,000 records per second
- **Memory usage**: Low memory footprint, suitable for large datasets
- **Accuracy**: High precision duplicate detection
- **Database coverage**: Scopus, PubMed, Web of Science, IEEE, CAB Abstracts, and more

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use this tool in your research, please cite:

```
Hakman, T. (2025). RIS deduplicator: Bibliographic record deduplication tool [Computer software]. 
https://github.com/yourusername/ris-deduplicator
```

## License

MIT License - see [LICENSE](LICENSE) file for details.
