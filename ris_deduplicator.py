#!/usr/bin/env python3
"""
RIS File Deduplicator

This script processes multiple RIS files, identifies duplicates based on key fields,
and outputs a single deduplicated RIS file.

The tool uses a similarity threshold (0.0-1.0) to control duplicate detection:
- 0.95+: Very strict (near-identical only)
- 0.85-0.94: Balanced approach (recommended) 
- 0.75-0.84: More sensitive (catches more duplicates)
- Below 0.75: Loose matching (high false positive risk)

Usage:
    python3 ris_deduplicator.py file1.ris file2.ris [--threshold 0.85] [-o output.ris]
    
Examples:
    python3 ris_deduplicator.py *.ris
    python3 ris_deduplicator.py *.ris --threshold 0.9  # Stricter matching
    python3 ris_deduplicator.py CAB-2.ris Sco-5.ris WoS-7.ris -o clean_bibliography.ris
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Callable, Any
import json

__version__ = "1.4.0"
__author__ = "RIS Deduplicator Contributors"

# Action types for duplicate handling
class ActionType:
    STATS = "stats"      # Add similarity scores to records
    MARK = "mark"        # Mark duplicates with field
    DELETE = "delete"    # Remove duplicates from output
    MERGE = "merge"      # Merge duplicate information (default)

# Field mutators for preprocessing
class FieldMutators:
    """Collection of field preprocessing functions inspired by dedupe-sweep."""
    
    @staticmethod
    def doi_normalize(value: str) -> str:
        """Normalize DOI by removing URL prefixes and lowercasing."""
        if not value:
            return ""
        doi = value.lower().strip()
        doi = re.sub(r'^https?://dx\.doi\.org/', '', doi)
        doi = re.sub(r'^https?://doi\.org/', '', doi)
        doi = re.sub(r'^doi:', '', doi)
        return doi
    
    @staticmethod
    def title_normalize(value: str, strict: bool = False) -> str:
        """Advanced title normalization."""
        if not value:
            return ""
        
        title = value.lower().strip()
        
        if not strict:
            # Remove common prefixes/suffixes
            title = re.sub(r'^(the|a|an)\s+', '', title)
            title = re.sub(r'\s+(the|a|an)$', '', title)
            
            # Normalize common variations
            title = re.sub(r'\bu\.?s\.?a?\b', 'usa', title)
            title = re.sub(r'\bu\.?k\.?\b', 'uk', title)
            title = re.sub(r'\bco2\b', 'carbon dioxide', title)
            title = re.sub(r'\bh2o\b', 'water', title)
            title = re.sub(r'\bn\.?a\.?\b', 'north america', title)
            title = re.sub(r'\be\.?u\.?\b', 'european union', title)
            
            # Remove punctuation except hyphens
            title = re.sub(r'[^\w\s\-]', ' ', title)
            title = re.sub(r'\s+', ' ', title)
        else:
            # Keep more structure for strict matching
            title = re.sub(r'[^\w\s\-\.]', ' ', title)
            title = re.sub(r'\s+', ' ', title)
        
        return title.strip()
    
    @staticmethod
    def author_normalize(value: str) -> str:
        """Normalize author names."""
        if not value:
            return ""
        
        author = value.lower().strip()
        # Remove common suffixes
        author = re.sub(r'\s+(jr\.?|sr\.?|ph\.?d\.?|m\.?d\.?)$', '', author)
        # Standardize spacing
        author = re.sub(r'\s+', ' ', author)
        return author.strip()
    
    @staticmethod
    def journal_normalize(value: str) -> str:
        """Normalize journal names."""
        if not value:
            return ""
        
        journal = value.lower().strip()
        # Remove common words
        journal = re.sub(r'\b(journal|of|the|and|international|american|european|proceedings)\b', '', journal)
        # Remove punctuation
        journal = re.sub(r'[^\w\s]', ' ', journal)
        journal = re.sub(r'\s+', ' ', journal)
        return journal.strip()
    
    @staticmethod
    def year_normalize(value: str) -> str:
        """Extract and normalize year."""
        if not value:
            return ""
        
        year_match = re.search(r'\d{4}', value)
        return year_match.group() if year_match else ""

# Comparison functions library
class ComparisonFunctions:
    """Collection of comparison functions inspired by dedupe-sweep."""
    
    @staticmethod
    def exact(a: str, b: str) -> float:
        """Exact string comparison."""
        if not a or not b:
            return 0.0
        return 1.0 if a == b else 0.0
    
    @staticmethod
    def jaccard(a: str, b: str) -> float:
        """Jaccard similarity of words."""
        if not a or not b:
            return 0.0
        
        words_a = set(a.split())
        words_b = set(b.split())
        
        if not words_a or not words_b:
            return 0.0
        
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def levenshtein_ratio(a: str, b: str) -> float:
        """Levenshtein distance as similarity ratio."""
        if not a or not b:
            return 0.0
        
        if a == b:
            return 1.0
        
        # Simple Levenshtein implementation
        len_a, len_b = len(a), len(b)
        if len_a > len_b:
            a, b = b, a
            len_a, len_b = len_b, len_a
        
        if len_a == 0:
            return 0.0
        
        # Create distance matrix
        distances = list(range(len_a + 1))
        
        for i in range(1, len_b + 1):
            new_distances = [i]
            for j in range(1, len_a + 1):
                cost = 0 if a[j-1] == b[i-1] else 1
                new_distances.append(min(
                    distances[j] + 1,      # deletion
                    new_distances[j-1] + 1, # insertion
                    distances[j-1] + cost   # substitution
                ))
            distances = new_distances
        
        max_len = max(len_a, len_b)
        return (max_len - distances[-1]) / max_len
    
    @staticmethod
    def fuzzy_match(a: str, b: str, threshold: float = 0.8) -> float:
        """Fuzzy matching combining multiple methods."""
        if not a or not b:
            return 0.0
        
        # Combine Jaccard and Levenshtein
        jaccard_score = ComparisonFunctions.jaccard(a, b)
        levenshtein_score = ComparisonFunctions.levenshtein_ratio(a, b)
        
        # Weighted average
        combined_score = (jaccard_score * 0.6) + (levenshtein_score * 0.4)
        
        return combined_score if combined_score >= threshold else 0.0
    
    @staticmethod
    def year_tolerance(a: str, b: str, tolerance: int = 1) -> float:
        """Year comparison with tolerance."""
        if not a or not b:
            return 0.0
        
        try:
            year_a = int(a)
            year_b = int(b)
            diff = abs(year_a - year_b)
            
            if diff == 0:
                return 1.0
            elif diff <= tolerance:
                return 0.8
            else:
                return 0.0
        except ValueError:
            return 0.0

# Deduplication strategies inspired by dedupe-sweep
class DeduplicationStrategies:
    """Collection of deduplication strategies with configurable steps."""
    
    @staticmethod
    def get_strategies() -> Dict[str, Dict]:
        """Return all available strategies."""
        return {
            'doi_only': {
                'title': 'DOI Only',
                'description': 'Compare references against DOI fields only',
                'mutators': {
                    'doi': 'doi_normalize'
                },
                'steps': [
                    {
                        'fields': ['doi'],
                        'comparison': 'exact',
                        'weight': 1.0,
                        'skip_omitted': True
                    }
                ]
            },
            'title_author_year': {
                'title': 'Title + Author + Year',
                'description': 'Traditional matching on title, first author, and year',
                'mutators': {
                    'title': 'title_normalize',
                    'authors': 'author_normalize',
                    'year': 'year_normalize'
                },
                'steps': [
                    {
                        'fields': ['title'],
                        'comparison': 'jaccard',
                        'weight': 0.5,
                        'skip_omitted': True
                    },
                    {
                        'fields': ['authors'],
                        'comparison': 'fuzzy_match',
                        'weight': 0.3,
                        'skip_omitted': True
                    },
                    {
                        'fields': ['year'],
                        'comparison': 'year_tolerance',
                        'weight': 0.2,
                        'skip_omitted': True
                    }
                ]
            },
            'comprehensive': {
                'title': 'Comprehensive Matching',
                'description': 'Multi-criteria matching including DOI, title, authors, year, and journal',
                'mutators': {
                    'doi': 'doi_normalize',
                    'title': 'title_normalize',
                    'authors': 'author_normalize',
                    'year': 'year_normalize',
                    'journal': 'journal_normalize'
                },
                'steps': [
                    {
                        'fields': ['doi'],
                        'comparison': 'exact',
                        'weight': 1.0,
                        'skip_omitted': True,
                        'exclusive': True  # If DOI matches, it's definitely the same paper
                    },
                    {
                        'fields': ['title'],
                        'comparison': 'fuzzy_match',
                        'weight': 0.4,
                        'skip_omitted': True
                    },
                    {
                        'fields': ['authors'],
                        'comparison': 'jaccard',
                        'weight': 0.3,
                        'skip_omitted': True
                    },
                    {
                        'fields': ['year'],
                        'comparison': 'year_tolerance',
                        'weight': 0.2,
                        'skip_omitted': True
                    },
                    {
                        'fields': ['journal'],
                        'comparison': 'jaccard',
                        'weight': 0.1,
                        'skip_omitted': True
                    }
                ]
            },
            'clark': {
                'title': 'Clark Method',
                'description': 'Based on the Clark deduplication method with fuzzy matching',
                'mutators': {
                    'title': 'title_normalize',
                    'authors': 'author_normalize',
                    'year': 'year_normalize',
                    'journal': 'journal_normalize'
                },
                'steps': [
                    {
                        'fields': ['title', 'authors'],
                        'comparison': 'fuzzy_match',
                        'weight': 0.6,
                        'skip_omitted': False
                    },
                    {
                        'fields': ['year'],
                        'comparison': 'year_tolerance',
                        'weight': 0.2,
                        'skip_omitted': True
                    },
                    {
                        'fields': ['journal'],
                        'comparison': 'fuzzy_match',
                        'weight': 0.2,
                        'skip_omitted': True
                    }
                ]
            }
        }


class RISRecord:
    """Represents a single RIS record with its fields."""
    
    def __init__(self):
        self.fields = defaultdict(list)
        self.raw_lines = []
    
    def add_field(self, tag: str, value: str):
        """Add a field to the record."""
        self.fields[tag].append(value.strip())
    
    def get_field(self, tag: str, default: str = "") -> str:
        """Get the first value of a field, or default if not present."""
        return self.fields[tag][0] if self.fields[tag] else default
    
    def get_all_fields(self, tag: str) -> List[str]:
        """Get all values for a field."""
        return self.fields[tag]
    
    def get_title(self) -> str:
        """Get the title, checking both TI and T1 fields."""
        return self.get_field('TI') or self.get_field('T1')
    
    def get_authors(self) -> List[str]:
        """Get all authors from AU and A1 fields."""
        authors = []
        authors.extend(self.get_all_fields('AU'))
        authors.extend(self.get_all_fields('A1'))
        return authors
    
    def get_doi(self) -> str:
        """Get DOI, cleaning common prefixes."""
        doi = self.get_field('DO')
        return FieldMutators.doi_normalize(doi)
    
    def get_year(self) -> str:
        """Get publication year from PY or Y1 fields."""
        year = self.get_field('PY') or self.get_field('Y1')
        return FieldMutators.year_normalize(year)
    
    def get_normalized_title(self, strict: bool = False) -> str:
        """Get normalized title for comparison using new mutator system."""
        title = self.get_title()
        return FieldMutators.title_normalize(title, strict)
    
    def get_author_signatures(self) -> List[str]:
        """Get normalized author signatures for comparison."""
        authors = self.get_authors()
        signatures = []
        
        for author in authors[:3]:  # Consider first 3 authors
            normalized = FieldMutators.author_normalize(author)
            
            # Extract last name
            if ',' in normalized:
                last_name = normalized.split(',')[0].strip()
            else:
                parts = normalized.split()
                last_name = parts[-1] if parts else ""
            
            # Clean up last name
            last_name = re.sub(r'[^\w]', '', last_name)
            if len(last_name) >= 2:  # Only include substantial names
                signatures.append(last_name)
        
        return signatures
    
    def get_journal_signature(self) -> str:
        """Get normalized journal/publication signature."""
        journal = self.get_field('T2') or self.get_field('JF') or self.get_field('JO')
        return FieldMutators.journal_normalize(journal)
    
    def get_duplicate_key(self, strict: bool = False) -> Tuple[str, str, str]:
        """Generate a key for duplicate detection based on normalized title, first author, and year."""
        title = self.get_normalized_title(strict)
        authors = self.get_author_signatures()
        first_author = authors[0] if authors else ""
        year = self.get_year()
        
        return (title, first_author, year)
    
    def calculate_comprehensive_similarity(self, other: 'RISRecord') -> float:
        """CONSERVATIVE similarity calculation - prioritizes avoiding false positives."""
        
        # DOI matching (highest priority - if different DOIs, definitely different papers)
        doi1, doi2 = self.get_doi(), other.get_doi()
        if doi1 and doi2:
            if doi1.lower() == doi2.lower():
                return 1.0  # Perfect match
            else:
                return 0.0  # Different DOIs = different papers
        
        # CONSERVATIVE APPROACH: All factors must be present and strong
        title1 = FieldMutators.title_normalize(self.get_title())
        title2 = FieldMutators.title_normalize(other.get_title())
        authors1 = self.get_authors()
        authors2 = other.get_authors()
        year1, year2 = self.get_year(), other.get_year()
        
        # STRICT REQUIREMENT: Must have title AND authors
        if not (title1 and title2 and authors1 and authors2):
            return 0.0  # Insufficient data for comparison
        
        # Title similarity - BALANCED
        title_sim = ComparisonFunctions.fuzzy_match(title1, title2, 0.75)  # Balanced threshold
        if title_sim < 0.65:  # Titles must be reasonably similar
            return 0.0
        
        # Author similarity - BALANCED
        author_sim = self._calculate_conservative_author_similarity(authors1, authors2)
        if author_sim < 0.6:  # Authors must have reasonable similarity
            return 0.0
        
        # Year matching - CONSERVATIVE (must be present and close)
        year_sim = 0.0
        if year1 and year2:
            year_sim = ComparisonFunctions.year_tolerance(year1, year2, 1)  # Only 1 year tolerance
        elif not year1 or not year2:
            year_sim = 0.3  # PENALTY for missing year data
        
        # WEIGHTED SCORE - all factors must be strong
        final_score = (title_sim * 0.6) + (author_sim * 0.3) + (year_sim * 0.1)
        
        # FINAL GATE: Must exceed high confidence threshold
        return final_score if final_score >= 0.75 else 0.0
    
    def _calculate_conservative_author_similarity(self, authors1: List[str], authors2: List[str]) -> float:
        """CONSERVATIVE author similarity - avoids false positives from common names."""
        if not authors1 or not authors2:
            return 0.0
        
        # Normalize authors
        norm_authors1 = [FieldMutators.author_normalize(a) for a in authors1[:2]]  # Only first 2
        norm_authors2 = [FieldMutators.author_normalize(a) for a in authors2[:2]]
        
        # Check for exact matches first (highest confidence)
        exact_matches = 0
        total_comparisons = 0
        
        for auth1 in norm_authors1:
            for auth2 in norm_authors2:
                total_comparisons += 1
                
                # Extract names
                last1 = self._extract_last_name(auth1)
                last2 = self._extract_last_name(auth2)
                first1 = self._extract_first_name(auth1)
                first2 = self._extract_first_name(auth2)
                
                if last1 and last2 and last1 == last2:
                    # CONSERVATIVE: Common surnames need stronger evidence
                    if self._is_common_surname(last1):
                        # Common surname - need exact first name/initial match
                        if first1 and first2 and (
                            first1 == first2 or 
                            self._initials_match(first1, first2)
                        ):
                            exact_matches += 1
                    else:
                        # Uncommon surname - last name match is strong evidence
                        if not first1 or not first2 or self._initials_match(first1, first2):
                            exact_matches += 1
        
        return exact_matches / total_comparisons if total_comparisons > 0 else 0.0
    
    def _is_common_surname(self, surname: str) -> bool:
        """Check if surname is common (needs stricter matching)."""
        common_surnames = {
            'wang', 'li', 'zhang', 'liu', 'chen', 'yang', 'huang', 'zhao', 'wu', 'zhou',
            'smith', 'johnson', 'williams', 'brown', 'jones', 'garcia', 'miller', 'davis',
            'rodriguez', 'martinez', 'hernandez', 'lopez', 'gonzalez', 'wilson', 'anderson',
            'thomas', 'taylor', 'moore', 'jackson', 'martin', 'lee', 'kim', 'park'
        }
        return surname.lower() in common_surnames
    
    def _extract_last_name(self, author: str) -> str:
        """Extract last name from author string."""
        if ',' in author:
            return author.split(',')[0].strip().lower()
        else:
            parts = author.split()
            return parts[-1].lower() if parts else ""
    
    def _extract_first_name(self, author: str) -> str:
        """Extract first name from author string."""
        if ',' in author:
            return author.split(',')[1].strip().lower() if ',' in author else ""
        else:
            parts = author.split()
            return ' '.join(parts[:-1]).lower() if len(parts) > 1 else ""
    
    def _initials_match(self, name1: str, name2: str) -> bool:
        """Check if names match as initials (e.g., 'h.-t.' matches 'hai-tao')."""
        if not name1 or not name2:
            return False
        
        # Extract initials from both names
        initials1 = ''.join([c for c in name1 if c.isalpha()]).lower()
        initials2 = ''.join([c for c in name2 if c.isalpha()]).lower()
        
        # Check if one is initials of the other
        if len(initials1) <= 3 and len(initials2) > 3:
            return initials2.startswith(initials1)
        elif len(initials2) <= 3 and len(initials1) > 3:
            return initials1.startswith(initials2)
        
        return initials1 == initials2

    def calculate_similarity_score(self, other: 'RISRecord') -> float:
        """Calculate similarity score between two records (0.0 to 1.0)."""
        score = 0.0
        factors = 0
        
        # DOI matching (highest priority)
        doi1, doi2 = self.get_doi(), other.get_doi()
        if doi1 and doi2:
            factors += 1
            if doi1.lower() == doi2.lower():
                score += 1.0
            else:
                return 0.0  # Different DOIs = definitely different papers
        
        # Title similarity
        title1 = self.get_normalized_title()
        title2 = other.get_normalized_title()
        if title1 and title2:
            factors += 1
            title_sim = self._string_similarity(title1, title2)
            score += title_sim
        
        # Author similarity
        authors1 = set(self.get_author_signatures())
        authors2 = set(other.get_author_signatures())
        if authors1 and authors2:
            factors += 1
            # Jaccard similarity for authors
            intersection = len(authors1 & authors2)
            union = len(authors1 | authors2)
            author_sim = intersection / union if union > 0 else 0.0
            score += author_sim
        
        # Year matching
        year1, year2 = self.get_year(), other.get_year()
        if year1 and year2:
            factors += 1
            if year1 == year2:
                score += 1.0
            elif abs(int(year1) - int(year2)) <= 1:  # Allow 1 year difference
                score += 0.8
        
        # Journal similarity
        journal1 = self.get_journal_signature()
        journal2 = other.get_journal_signature()
        if journal1 and journal2:
            factors += 1
            journal_sim = self._string_similarity(journal1, journal2)
            score += journal_sim * 0.5  # Lower weight for journal
        
        return score / factors if factors > 0 else 0.0
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity using Jaccard similarity of words."""
        if not s1 or not s2:
            return 0.0
        
        words1 = set(s1.split())
        words2 = set(s2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def merge_with(self, other: 'RISRecord') -> 'RISRecord':
        """Merge this record with another, keeping the most complete information."""
        merged = RISRecord()
        
        # Combine all fields, preferring non-empty values
        all_tags = set(self.fields.keys()) | set(other.fields.keys())
        
        for tag in all_tags:
            self_values = self.get_all_fields(tag)
            other_values = other.get_all_fields(tag)
            
            # Combine values, removing duplicates while preserving order
            combined = []
            seen = set()
            for value in self_values + other_values:
                if value and value not in seen:
                    combined.append(value)
                    seen.add(value)
            
            for value in combined:
                merged.add_field(tag, value)
        
        return merged
    
    def to_ris_string(self) -> str:
        """Convert the record back to RIS format."""
        lines = []
        
        # Start with TY field
        if 'TY' in self.fields:
            lines.append(f"TY  - {self.fields['TY'][0]}")
        
        # Add other fields in a consistent order
        field_order = ['AU', 'A1', 'TI', 'T1', 'PY', 'Y1', 'T2', 'JF', 'VL', 'IS', 'SP', 'EP', 
                      'DO', 'UR', 'AB', 'N2', 'KW', 'DB', 'ID', 'SN', 'PB', 'CY', 'LA', 'PT', 'M1', 'M3', 'AD', 'C3', 'CR', 'C7']
        
        for tag in field_order:
            if tag in self.fields and tag != 'TY':
                for value in self.fields[tag]:
                    lines.append(f"{tag}  - {value}")
        
        # Add any remaining fields not in the ordered list
        for tag, values in self.fields.items():
            if tag not in field_order and tag != 'TY':
                for value in values:
                    lines.append(f"{tag}  - {value}")
        
        lines.append("ER  -")
        return '\n'.join(lines)


def parse_ris_file(filename: str) -> List[RISRecord]:
    """Parse a RIS file and return a list of RISRecord objects."""
    records = []
    current_record = None
    line_num = 0
    
    # Try different encodings if UTF-8 fails
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'ascii']
    
    for encoding in encodings:
        try:
            with open(filename, 'r', encoding=encoding, errors='replace') as file:
                for line_num, line in enumerate(file, 1):
                    line = line.rstrip('\n\r')
                    
                    # Skip empty lines at the start before any record content
                    if not line.strip() and current_record is None:
                        continue
                    
                    # Check if this is a field line (flexible spacing around dash)
                    field_match = re.match(r'^([A-Z0-9]{2})\s*-\s*(.*)$', line)
                    
                    if field_match:
                        tag, value = field_match.groups()
                        
                        if tag == 'TY':
                            # Start of new record with TY marker
                            if current_record:
                                records.append(current_record)
                            current_record = RISRecord()
                        elif current_record is None:
                            # Start of a record without TY marker (create record automatically)
                            current_record = RISRecord()
                        
                        if current_record:
                            current_record.add_field(tag, value)
                    
                    elif re.match(r'^ER\s*-\s*$', line.strip()):
                        # End of record (flexible spacing)
                        if current_record:
                            records.append(current_record)
                            current_record = None
                        # Note: Some files may have orphaned ER lines, which we ignore
                    
                    # Handle continuation lines (lines that don't match field pattern but aren't empty)
                    elif current_record and line.strip() and not line.startswith('  '):
                        # This might be a continuation of the previous field or malformed line
                        # For robustness, we'll skip it with a warning for very malformed lines
                        if len(line) > 100:  # Probably a data line that got mangled
                            print(f"Warning: Skipping potentially malformed line {line_num} in {os.path.basename(filename)}")
                
                # Don't forget the last record if file doesn't end with ER
                if current_record:
                    records.append(current_record)
                
                break  # Successfully parsed with this encoding
                
        except UnicodeDecodeError:
            continue  # Try next encoding
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found")
            return []
        except PermissionError:
            print(f"Error: Permission denied reading '{filename}'")
            return []
        except Exception as e:
            print(f"Error reading {filename} with encoding {encoding}: {e}")
            if encoding == encodings[-1]:  # Last encoding failed
                return []
            continue
    
    if not records:
        print(f"Warning: No valid records found in {os.path.basename(filename)}")
    
    return records


def find_duplicates_accurate(records: List[RISRecord], similarity_threshold: float = 0.85) -> Dict[int, List[RISRecord]]:
    """Accurate duplicate detection - balanced precision and recall."""
    
    # First, group by DOI (exact matches only)
    doi_groups = defaultdict(list)
    no_doi_records = []
    
    for record in records:
        doi = record.get_doi()
        if doi and doi.strip():  # Must have non-empty DOI
            doi_groups[doi.lower().strip()].append(record)
        else:
            no_doi_records.append(record)
    
    # DOI-based groups (high confidence)
    duplicate_groups = {}
    group_id = 0
    
    for doi, group in doi_groups.items():
        if len(group) > 1:
            duplicate_groups[group_id] = group
            group_id += 1
    
    # CONSERVATIVE non-DOI matching - NO TRANSITIVE GROUPING
    processed = set()
    
    for i, record1 in enumerate(no_doi_records):
        if i in processed:
            continue
            
        # Find DIRECT matches only (no chaining)
        direct_matches = [record1]
        
        for j, record2 in enumerate(no_doi_records[i+1:], i+1):
            if j in processed:
                continue
            
            # Calculate similarity
            similarity = record1.calculate_comprehensive_similarity(record2)
            
            # BALANCED THRESHOLD
            if similarity >= max(similarity_threshold, 0.8):  # At least 0.8
                direct_matches.append(record2)
                processed.add(j)
        
        # Only create group if we found direct matches
        if len(direct_matches) > 1:
            # ADDITIONAL VALIDATION: Check if all records are actually similar to each other
            if len(direct_matches) <= 3 or all_mutually_similar(direct_matches, similarity_threshold):
                duplicate_groups[group_id] = direct_matches
                group_id += 1
                processed.add(i)
    
    return duplicate_groups

def all_mutually_similar(records: List[RISRecord], threshold: float) -> bool:
    """Verify all records in group are mutually similar (prevents chaining false positives)."""
    if len(records) <= 2:
        return True
    
    # Check all pairs
    for i in range(len(records)):
        for j in range(i+1, len(records)):
            similarity = records[i].calculate_comprehensive_similarity(records[j])
            if similarity < threshold:
                return False
    return True


def deduplicate_records_advanced(records: List[RISRecord], similarity_threshold: float = 0.8) -> Tuple[List[RISRecord], Dict[int, List[RISRecord]]]:
    """Advanced deduplication using improved algorithm."""
    duplicate_groups = find_duplicates_accurate(records, similarity_threshold)
    
    # Keep track of which records are duplicates
    duplicate_records = set()
    for group in duplicate_groups.values():
        for record in group[1:]:  # Keep first, mark others as duplicates
            duplicate_records.add(id(record))
    
    # Merge duplicate groups
    merged_records = []
    processed_groups = set()
    
    for record in records:
        record_id = id(record)
        
        # Check if this record is part of a duplicate group
        found_group = None
        for group_id, group in duplicate_groups.items():
            if any(id(r) == record_id for r in group):
                found_group = group_id
                break
        
        if found_group is not None and found_group not in processed_groups:
            # This is the first record in a duplicate group - merge all duplicates
            group = duplicate_groups[found_group]
            merged_record = group[0]
            for duplicate in group[1:]:
                merged_record = merged_record.merge_with(duplicate)
            merged_records.append(merged_record)
            processed_groups.add(found_group)
        elif record_id not in duplicate_records:
            # This is not a duplicate
            merged_records.append(record)
    
    return merged_records, duplicate_groups


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Smart RIS bibliography deduplicator that identifies and merges duplicate academic records using DOI matching and intelligent similarity analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s *.ris                                    # Process all RIS files (default threshold 0.85)
  %(prog)s database1.ris database2.ris database3.ris  # Merge specific files
  %(prog)s *.ris -o clean_references.ris           # Custom output filename
  %(prog)s *.ris --threshold 0.9 --show-duplicates # High precision with details
  %(prog)s *.ris --threshold 0.75 --verbose        # More sensitive duplicate detection
  %(prog)s *.ris --threshold 0.95                  # Very conservative (near-identical only)

Threshold Selection Guide:
  • First run: Use default (0.85) - good for most academic databases
  • Too few duplicates found: Lower threshold (0.75-0.80) to be more sensitive
  • Too many false positives: Raise threshold (0.90-0.95) to be more selective
  • High-quality databases: Higher threshold (0.90+) for precision
  • Mixed/dirty data: Lower threshold (0.75-0.80) but review results carefully

The tool intelligently identifies duplicates by:
  • Exact DOI matching (highest confidence, bypasses threshold)
  • Title similarity analysis with fuzzy matching
  • Author name comparison with normalization
  • Publication year tolerance (±1 year)
        """
    )
    
    parser.add_argument(
        'input_files',
        nargs='+',
        help='RIS files to merge and deduplicate (supports wildcards like *.ris)'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='merged_deduplicated.ris',
        help='Output filename for deduplicated results (default: merged_deduplicated.ris)'
    )
    
    parser.add_argument(
        '--show-duplicates',
        action='store_true',
        help='Display detailed information about each duplicate group found'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed progress and statistics during processing'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.85,
        help='Similarity threshold (0.0-1.0): 0.95+ = very strict, 0.85-0.94 = balanced (recommended), 0.75-0.84 = relaxed, <0.75 = loose. Default: 0.85'
    )
    
    return parser.parse_args()


def validate_input_files(input_files):
    """Validate that input files exist and are readable."""
    valid_files = []
    for filename in input_files:
        if not os.path.exists(filename):
            print(f"Warning: File '{filename}' does not exist, skipping...")
            continue
        if not os.path.isfile(filename):
            print(f"Warning: '{filename}' is not a file, skipping...")
            continue
        if not filename.lower().endswith('.ris'):
            print(f"Warning: '{filename}' does not have .ris extension, but will process anyway...")
        valid_files.append(filename)
    
    return valid_files


def show_duplicate_details(duplicate_groups, algorithm='legacy'):
    """Show detailed information about duplicate groups."""
    print("\nDuplicate Groups Found:")
    print("=" * 80)
    
    if algorithm in ['advanced', 'strategy']:
        # Handle new format with numeric group IDs
        for group_id, group in duplicate_groups.items():
            print(f"\nGroup {group_id + 1} ({len(group)} records):")
            
            # Show similarity scores between records in the group
            if len(group) > 1:
                first_record = group[0]
                print(f"  Title: {first_record.get_title()[:80]}{'...' if len(first_record.get_title()) > 80 else ''}")
                print(f"  Main Author: {first_record.get_authors()[0] if first_record.get_authors() else 'N/A'}")
                print(f"  Year: {first_record.get_year()}")
                
                # Show records with similarity scores
                for j, record in enumerate(group):
                    doi = record.get_doi()
                    db = record.get_field('DB')
                    if j > 0:
                        similarity = first_record.calculate_similarity_score(record)
                        print(f"    Record {j+1}: DB={db}, DOI={doi[:40]}{'...' if len(doi) > 40 else ''}, Similarity: {similarity:.3f}")
                    else:
                        print(f"    Record {j+1}: DB={db}, DOI={doi[:40]}{'...' if len(doi) > 40 else ''} (reference)")
    else:
        # Handle legacy format with tuple keys
        for i, (key, group) in enumerate(duplicate_groups.items(), 1):
            title, author, year = key
            print(f"\nGroup {i} ({len(group)} records):")
            print(f"  Title: {title[:80]}{'...' if len(title) > 80 else ''}")
            print(f"  Author: {author}")
            print(f"  Year: {year}")
            
            for j, record in enumerate(group):
                doi = record.get_doi()
                db = record.get_field('DB')
                print(f"    Record {j+1}: DB={db}, DOI={doi[:50]}{'...' if len(doi) > 50 else ''}")


def main():
    """Main function to process RIS files and output deduplicated result."""
    args = parse_arguments()
    
    # Validate input files
    input_files = validate_input_files(args.input_files)
    
    if not input_files:
        print("❌ Error: No valid input files found.")
        sys.exit(1)
    
    # Validate threshold
    if not 0.0 <= args.threshold <= 1.0:
        print("❌ Error: Threshold must be between 0.0 and 1.0")
        sys.exit(1)
    
    if args.verbose:
        print(f"🔧 Running RIS Deduplicator v{__version__}")
        
        # Provide threshold guidance
        if args.threshold >= 0.95:
            threshold_desc = "(very strict - near-identical records only)"
        elif args.threshold >= 0.9:
            threshold_desc = "(strict - high precision, may miss some duplicates)"
        elif args.threshold >= 0.85:
            threshold_desc = "(balanced - recommended for most cases)"
        elif args.threshold >= 0.75:
            threshold_desc = "(relaxed - more sensitive, may include false positives)"
        else:
            threshold_desc = "(loose - high false positive risk, review results carefully)"
            
        print(f"🔧 Similarity threshold: {args.threshold} {threshold_desc}")
    
    print(f"📂 Processing {len(input_files)} RIS file(s)...")
    if args.verbose:
        print("Input files:")
        for f in input_files:
            print(f"  - {f}")
        print(f"Output file: {args.output}")
        print()
    
    # Read all files
    print("📖 Reading RIS files...")
    all_records = []
    file_stats = {}
    
    for filename in input_files:
        base_name = os.path.basename(filename)
        if args.verbose:
            print(f"  Reading {base_name}...")
        else:
            print(f"  Reading {base_name}...", end=' ', flush=True)
        
        records = parse_ris_file(filename)
        if records:
            file_stats[base_name] = len(records)
            if args.verbose:
                print(f"    Found {len(records)} records")
            else:
                print(f"({len(records)} records)")
            all_records.extend(records)
        else:
            file_stats[base_name] = 0
            print(f"    ⚠️  Warning: No records found or error reading file")
    
    if not all_records:
        print("❌ Error: No records found in any input files.")
        print("   Make sure your files are valid RIS format with TY and ER markers.")
        sys.exit(1)
    
    print(f"\n📊 Total records read: {len(all_records)}")
    
    # Find duplicates using balanced precision and recall
    print("🔍 Finding duplicates...")
    duplicate_groups = find_duplicates_accurate(all_records, args.threshold)
    total_duplicates = sum(len(group) - 1 for group in duplicate_groups.values())
    
    if args.verbose or total_duplicates > 0:
        print(f"Found {len(duplicate_groups)} duplicate groups with {total_duplicates} duplicate records")
        
        if args.verbose and total_duplicates > 0:
            doi_groups = sum(1 for group in duplicate_groups.values() 
                           if any(record.get_doi() for record in group))
            print(f"  • {doi_groups} groups matched by DOI")
            print(f"  • {len(duplicate_groups) - doi_groups} groups matched by similarity")
    
    # Show duplicate details if requested
    if args.show_duplicates and duplicate_groups:
        show_duplicate_details(duplicate_groups, 'advanced')
    
    # Deduplicate
    print("🔧 Deduplicating records...")
    deduplicated_records, _ = deduplicate_records_advanced(all_records, args.threshold)
    print(f"After deduplication: {len(deduplicated_records)} unique records")
    
    # Write output
    print(f"\n💾 Writing deduplicated records to {args.output}...")
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            if args.verbose:
                print(f"  Created directory: {output_dir}")
        
        with open(args.output, 'w', encoding='utf-8') as f:
            for i, record in enumerate(deduplicated_records):
                if i > 0:
                    f.write('\n\n')  # Add spacing between records
                f.write(record.to_ris_string())
        
        print(f"✅ Successfully wrote {len(deduplicated_records)} deduplicated records to {args.output}")
        
        if total_duplicates > 0:
            reduction_pct = total_duplicates/len(all_records)*100
            print(f"🧹 Removed {total_duplicates} duplicate records ({reduction_pct:.1f}% reduction)")
            print(f"📊 Result: {len(all_records)} → {len(deduplicated_records)} unique records")
            
            if args.verbose:
                print(f"\n📈 Input summary:")
                for filename, count in file_stats.items():
                    print(f"  • {filename}: {count} records")
                print(f"\n📊 Deduplication summary:")
                print(f"  • Total input records:    {len(all_records)}")
                print(f"  • Unique records found:   {len(deduplicated_records)}")
                print(f"  • Duplicates removed:     {total_duplicates}")
                print(f"  • Space saved:            {reduction_pct:.1f}%")
        else:
            print("🎉 No duplicates found - all records are unique!")
            if args.threshold >= 0.9:
                print("💡 Your threshold is quite strict. Try --threshold 0.85 or 0.75 to find more potential duplicates")
            elif args.threshold >= 0.85:
                print("💡 Consider lowering to --threshold 0.75 if you expected to find duplicates")
            else:
                print("💡 Your data appears to have genuinely unique records")
        
    except PermissionError:
        print(f"❌ Error: Permission denied writing to '{args.output}'")
        print("   Make sure you have write access to the output directory.")
        sys.exit(1)
    except OSError as e:
        print(f"❌ Error writing output file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error writing output file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
