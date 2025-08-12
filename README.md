# RIS File Deduplicator

A smart Python tool to find and remove duplicates between RIS (Research Information Systems) files using intelligent similarity analysis and DOI matching. Perfect for researchers managing bibliographies from multiple databases.

**⚡ Key Feature**: Adjustable similarity threshold (0.0-1.0) lets you control precision vs sensitivity - from conservative matching (0.9+) to more inclusive detection (0.75-0.8).

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🚀 **Easy to use**: Simple command-line interface with clear guidance
- 📁 **Flexible input**: Process any number of RIS files with wildcard support
- 🧠 **Smart duplicate detection**: Intelligent similarity analysis with DOI priority matching
- 🎯 **Configurable precision**: Adjustable similarity threshold (0.0-1.0) with clear guidance
- 🔀 **Information merging**: Combines data from duplicate records to keep the most complete information
- 📋 **Multiple output options**: Choose your output filename and location
- 🔍 **Detailed reporting**: See exactly what duplicates were found with similarity scores
- 🛡️ **Robust parsing**: Handles different character encodings and malformed files
- 📊 **Smart feedback**: Context-aware suggestions and detailed statistics
- 🎯 **DOI-based matching**: Highest priority matching for records with DOIs (bypasses threshold)
- 🔬 **Multi-criteria scoring**: Title, authors, year, and journal similarity with fuzzy matching

## Installation

### Prerequisites
- Python 3.6 or higher
- No external dependencies required (uses only Python standard library)

### Download

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ris-deduplicator.git
   cd ris-deduplicator
   ```

2. **Or download the script directly:**
   ```bash
   curl -O https://raw.githubusercontent.com/yourusername/ris-deduplicator/main/ris_deduplicator.py
   ```

## Quick Start

```bash
# Process all RIS files in current directory (default threshold: 0.85)
python3 ris_deduplicator.py *.ris

# Process specific files with custom output
python3 ris_deduplicator.py database1.ris database2.ris database3.ris -o clean_bibliography.ris

# Show detailed duplicate information
python3 ris_deduplicator.py *.ris --show-duplicates

# Use verbose mode for detailed processing info
python3 ris_deduplicator.py *.ris --verbose

# Adjust similarity threshold for different precision levels
python3 ris_deduplicator.py *.ris --threshold 0.9   # Stricter (fewer false positives)
python3 ris_deduplicator.py *.ris --threshold 0.75  # More sensitive (finds more duplicates)
python3 ris_deduplicator.py *.ris --threshold 0.95  # Very conservative (near-identical only)
```

## Usage

```
python3 ris_deduplicator.py [input_files...] [-o output.ris] [--threshold 0.85] [--show-duplicates] [--verbose]
```

### Arguments

- `input_files`: One or more RIS files to process (supports wildcards like *.ris)
- `-o, --output`: Output filename (default: `merged_deduplicated.ris`)
- `--threshold`: Similarity threshold for duplicate detection (0.0-1.0, default: 0.85)
- `--show-duplicates`: Display detailed information about each duplicate group found
- `-v, --verbose`: Show detailed progress and statistics during processing
- `--version`: Show version information

### Similarity Threshold Guide

The `--threshold` parameter controls how similar records must be to be considered duplicates:

| Range | Description | Use Case | False Positives | False Negatives |
|-------|-------------|----------|----------------|----------------|
| **0.95-1.0** | Very strict | High-quality databases, final cleanup | Very low | Higher |
| **0.85-0.94** | Balanced ✅ | Most academic databases (recommended) | Low | Moderate |
| **0.75-0.84** | Relaxed | Mixed quality data, initial screening | Moderate | Lower |
| **0.50-0.74** | Loose | Very dirty data (review results carefully) | High | Very low |

**💡 Recommendations:**
- **First run**: Use default (0.85) - good balance for most cases
- **Too few duplicates found**: Lower to 0.75-0.80 for more sensitivity  
- **Too many false positives**: Raise to 0.90-0.95 for more precision
- **High-quality databases**: Use 0.90+ for precision
- **Mixed/dirty data**: Use 0.75-0.80 but review results

### Examples

```bash
# Basic usage - process all .ris files with default balanced threshold
python3 ris_deduplicator.py *.ris

# Specific files with custom output
python3 ris_deduplicator.py scopus.ris pubmed.ris web_of_science.ris -o bibliography.ris

# High precision for clean databases (stricter matching)
python3 ris_deduplicator.py *.ris --threshold 0.9 -o clean_refs.ris

# More sensitive detection for mixed quality data
python3 ris_deduplicator.py *.ris --threshold 0.75 --verbose

# See detailed duplicate information with balanced threshold
python3 ris_deduplicator.py *.ris --show-duplicates

# Very conservative matching (near-identical records only)
python3 ris_deduplicator.py *.ris --threshold 0.95

# Process files from different directories with custom threshold
python3 ris_deduplicator.py /path/to/file1.ris /path/to/file2.ris --threshold 0.8 -o /output/clean.ris
```

## How It Works

The tool uses a sophisticated multi-layered approach to identify duplicates:

1. **File Parsing**: Reads all RIS files with robust encoding detection
2. **DOI Priority Matching**: 
   - Records with identical DOIs are automatically grouped as duplicates
   - Different DOIs = definitely different papers (bypasses similarity threshold)
3. **Intelligent Similarity Analysis** for non-DOI records:
   - **Title similarity**: Fuzzy matching with advanced normalization
   - **Author comparison**: Multi-author similarity using last names and initials
   - **Year tolerance**: Allows ±1 year publication discrepancies
   - **Journal matching**: Publication venue similarity (optional weight)
4. **Conservative Grouping**: Prevents false positives through mutual similarity verification
5. **Smart Merging**: Combines information from duplicate records, preserving all available data
6. **Clean Output**: Writes a deduplicated RIS file maintaining standard formatting

## Sample Output

### Default Run (Threshold 0.85)
```
🔧 Running RIS Deduplicator v1.4.0
🔧 Similarity threshold: 0.85 (balanced - recommended for most cases)
📂 Processing 3 RIS file(s)...
📖 Reading RIS files...
  Reading CAB-2.ris... (682 records)
  Reading Sco-5.ris... (994 records)  
  Reading WoS-7.ris... (630 records)

📊 Total records read: 2306
🔍 Finding duplicates...
Found 609 duplicate groups with 960 duplicate records
  • 566 groups matched by DOI
  • 43 groups matched by similarity
🔧 Deduplicating records...
After deduplication: 1353 unique records

✅ Successfully wrote 1353 deduplicated records to merged_deduplicated.ris
🧹 Removed 960 duplicate records (41.6% reduction)
📊 Result: 2306 → 1353 unique records
```

### Strict Threshold (0.9) - Higher Precision
```
🔧 Similarity threshold: 0.9 (strict - high precision, may miss some duplicates)
📂 Processing 3 RIS file(s)...
📊 Total records read: 2306
🔍 Finding duplicates...
Found 580 duplicate groups with 890 duplicate records
  • 566 groups matched by DOI  
  • 14 groups matched by similarity
🧹 Removed 890 duplicate records (38.6% reduction)
```

### Relaxed Threshold (0.75) - Higher Sensitivity
```
🔧 Similarity threshold: 0.75 (relaxed - more sensitive, may include false positives)
📂 Processing 3 RIS file(s)...
📊 Total records read: 2306
🔍 Finding duplicates...
Found 645 duplicate groups with 1050 duplicate records
  • 566 groups matched by DOI
  • 79 groups matched by similarity
🧹 Removed 1050 duplicate records (45.5% reduction)
💡 Review similarity-matched groups carefully for false positives
```

### Threshold Comparison
| Threshold | Duplicates Found | Precision | Sensitivity | Best For |
|-----------|-----------------|-----------|-------------|----------|
| 0.95 | 850 (36.9%) | Very High | Lower | Final cleanup, high-quality databases |
| 0.85 | 960 (41.6%) | High | Moderate | Most academic databases ✅ |
| 0.75 | 1050 (45.5%) | Moderate | Higher | Mixed quality data, initial screening |

## Supported RIS Sources

This tool has been tested with RIS files from:
- **Scopus**
- **Web of Science**
- **PubMed**
- **CAB Abstracts**
- **IEEE Xplore**
- Most other databases that export standard RIS format

## Intelligent Duplicate Detection

The tool uses a sophisticated single-algorithm approach that balances accuracy and performance:

### Smart Detection Process

1. **DOI Priority Matching**: 
   - Records with identical DOIs are automatically considered duplicates
   - Different DOIs = definitely different papers (bypasses similarity threshold)
   - Provides highest confidence matching

2. **Multi-Criteria Similarity Analysis** for non-DOI records:
   - **Title similarity** (60% weight): Fuzzy matching with advanced normalization
   - **Author similarity** (30% weight): Conservative comparison using last names and initials  
   - **Year tolerance** (10% weight): Allows ±1 year publication discrepancies
   - **Journal matching**: Optional additional verification

3. **Advanced Field Preprocessing**:
   - Removes common words (the, a, an) and punctuation
   - Normalizes abbreviations (U.S.A → usa, CO2 → carbon dioxide) 
   - Handles author name variations and initials
   - Conservative approach for common surnames

4. **False Positive Prevention**:
   - Mutual similarity verification for groups
   - Conservative thresholds for common author names
   - Strict requirements for title and author presence

**Key Advantages**:
- Finds 35-45% duplicates in typical academic databases
- Adjustable precision via similarity threshold
- Conservative approach minimizes false positives
- Handles mixed-quality data effectively

## Technical Details

- **Memory efficient**: Processes large files without loading everything into memory
- **Encoding robust**: Automatically detects and handles different character encodings
- **Error resilient**: Continues processing even with malformed records
- **Format preservation**: Maintains all RIS fields and proper formatting

## Notes

- The script preserves all bibliographic information when merging duplicates
- Creates output directories automatically if they don't exist
- Handles various RIS field formats (TI/T1 for titles, AU/A1 for authors, etc.)
- Warns about missing files or non-RIS extensions but continues processing
- Output follows standard RIS format and can be imported into reference managers

## Common Use Cases

### Academic Researchers (Balanced Approach)
```bash
# Merge literature search results from multiple databases with default threshold
python3 ris_deduplicator.py scopus_search.ris pubmed_search.ris ieee_search.ris -o literature_review.ris

# More conservative approach for high-quality databases
python3 ris_deduplicator.py *.ris --threshold 0.9 -o conservative_refs.ris
```

### Systematic Reviews (Methodological Documentation)
```bash
# Process with detailed logging and moderate threshold for methodology documentation
python3 ris_deduplicator.py database_*.ris --threshold 0.85 -v --show-duplicates -o systematic_review_refs.ris

# Two-stage approach: first relaxed, then manual review
python3 ris_deduplicator.py database_*.ris --threshold 0.75 --show-duplicates -o stage1_screening.ris
```

### Large Scale Projects (Mixed Quality Data)
```bash
# Process mixed quality data with initial relaxed threshold
python3 ris_deduplicator.py /data/batch1/*.ris /data/batch2/*.ris --threshold 0.8 -o /results/merged.ris

# High-precision final cleanup
python3 ris_deduplicator.py /results/merged.ris --threshold 0.95 -o /results/final_clean.ris
```

### Database-Specific Recommendations
```bash
# High-quality databases (Scopus, Web of Science)
python3 ris_deduplicator.py *.ris --threshold 0.9

# Mixed databases or older data
python3 ris_deduplicator.py *.ris --threshold 0.75 --verbose

# Final cleanup after manual review
python3 ris_deduplicator.py *.ris --threshold 0.95
```

## Troubleshooting

### Common Issues

- **"No records found"**: Check that your RIS files are properly formatted and contain `TY  -` and `ER  -` markers
- **"File not found"**: Ensure file paths are correct and files exist  
- **"Permission denied"**: Make sure you have read access to input files and write access to output location
- **"Encoding errors"**: The tool automatically tries multiple encodings, but some very old files might need manual conversion

### Duplicate Detection Issues

- **"No duplicates found" (but you expected some)**:
  - Try lowering the threshold: `--threshold 0.75` or `--threshold 0.8`
  - Use `--verbose` to see what's happening during processing
  - Check if your records have DOIs (shown in verbose output)

- **"Too many false positives"**:
  - Raise the threshold: `--threshold 0.9` or `--threshold 0.95`
  - Use `--show-duplicates` to review what's being matched
  - Consider that records with different DOIs will never match

- **"Results seem inconsistent"**:
  - Records with DOIs bypass the similarity threshold completely
  - Title and author fields must be present for similarity matching
  - Year differences of >1 year reduce similarity scores significantly

### Performance Issues

- **"Processing is slow"**: The tool is optimized for accuracy over speed. For very large datasets (>10,000 records), consider processing in smaller batches
- **"Memory usage is high"**: The tool loads all records into memory. For extremely large files, split them first

### Getting Help

```bash
# Show all available options and threshold guidance
python3 ris_deduplicator.py --help

# Check version
python3 ris_deduplicator.py --version

# Test with verbose output to see what's happening and threshold feedback
python3 ris_deduplicator.py your_file.ris --verbose

# Test different thresholds to find the right balance
python3 ris_deduplicator.py your_file.ris --threshold 0.85 --show-duplicates
python3 ris_deduplicator.py your_file.ris --threshold 0.75 --show-duplicates
python3 ris_deduplicator.py your_file.ris --threshold 0.9 --show-duplicates
```

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup
```bash
git clone https://github.com/yourusername/ris-deduplicator.git
cd ris-deduplicator
# No additional setup needed - uses only Python standard library
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this tool in your research, please cite:
```
RIS Deduplicator (2024). GitHub repository: https://github.com/yourusername/ris-deduplicator
```
