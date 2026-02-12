#!/usr/bin/env python
"""
Axis-Based Realignment with Collision Text Merging

Reconstructs document text by actively realigning words along dynamic horizontal axes
and merging strongly colliding text boxes to remove OCR duplicates.

Algorithm:
1. Seed word selection
2. Axis creation
3. Axis extension and collision detection
4. Vertical realignment
5. Strong box collision detection
6. Text-level merge for strongly colliding boxes
7. Recursive axis propagation
8. Line reconstruction
9. Final output
"""

from typing import List, Dict, Set, Tuple, Optional
from pathlib import Path
import numpy as np
import yaml

_cfg = yaml.safe_load(open(Path('config/config.yaml')))

# Configuration — loaded from config.yaml
_recon_cfg = _cfg["reconstruction"]
RIGHT_MARGIN_RATIO = _recon_cfg.get("right_margin_ratio", 0.95)
SMALL_GAP_RATIO = _recon_cfg.get("small_gap_ratio", 0.3)
TAB_GAP_RATIO = _recon_cfg.get("tab_gap_ratio", 2.0)
STRONG_COLLISION_THRESHOLD = _recon_cfg.get("strong_collision_threshold", 0.30)



class Word:
    """Represents a word with modifiable geometry."""
    
    def __init__(self, text: str, bbox: List[float], word_id: int):
        self.text = text
        self.x0, self.y0, self.x1, self.y1 = bbox
        self.id = word_id
        self.aligned = False
        self.merged = False  # Marked for removal after merge
        self.axis_id = None
    
    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2
    
    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2
    
    @property
    def width(self) -> float:
        return self.x1 - self.x0
    
    @property
    def height(self) -> float:
        return self.y1 - self.y0
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def move_to_axis(self, axis_y: float):
        """Move word vertically so its center aligns with axis_y."""
        delta = axis_y - self.cy
        self.y0 += delta
        self.y1 += delta
    
    def intersects_axis(self, axis_y: float) -> bool:
        """Check if horizontal axis passes through this word's bbox."""
        return self.y0 <= axis_y <= self.y1
    
    def bbox_overlap_ratio(self, other: 'Word') -> float:
        """Compute overlap ratio (intersection / smaller box area)."""
        # Intersection
        ix0 = max(self.x0, other.x0)
        iy0 = max(self.y0, other.y0)
        ix1 = min(self.x1, other.x1)
        iy1 = min(self.y1, other.y1)
        
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        
        intersection = (ix1 - ix0) * (iy1 - iy0)
        smaller_area = min(self.area, other.area)
        
        if smaller_area <= 0:
            return 0.0
        
        return intersection / smaller_area
    
    def merge_with(self, other: 'Word') -> 'Word':
        """Merge this word with another, combining text and bbox."""
        # Merge text using longest common overlap
        merged_text = merge_overlapping_text(self.text, other.text)
        
        # Union of bounding boxes
        new_bbox = [
            min(self.x0, other.x0),
            min(self.y0, other.y0),
            max(self.x1, other.x1),
            max(self.y1, other.y1)
        ]
        
        # Create merged word (keep this word's ID)
        merged = Word(merged_text, new_bbox, self.id)
        merged.aligned = self.aligned
        merged.axis_id = self.axis_id
        
        return merged


def find_longest_overlap(s1: str, s2: str) -> int:
    """
    Find longest overlap where end of s1 matches beginning of s2.
    Returns length of overlap.
    """
    max_overlap = min(len(s1), len(s2))
    
    for i in range(max_overlap, 0, -1):
        if s1[-i:] == s2[:i]:
            return i
    
    return 0


def merge_overlapping_text(text1: str, text2: str) -> str:
    """
    Merge two text strings by finding overlapping substring.
    Example: "abcdef" + "defghijk" → "abcdefghijk"
    """
    if not text1:
        return text2
    if not text2:
        return text1
    
    # Check if text2 is contained in text1
    if text2 in text1:
        return text1
    
    # Check if text1 is contained in text2
    if text1 in text2:
        return text2
    
    # Find overlap at end of text1 / beginning of text2
    overlap_len = find_longest_overlap(text1, text2)
    
    if overlap_len > 0:
        return text1 + text2[overlap_len:]
    
    # No overlap - just concatenate with space
    return text1 + " " + text2


class Axis:
    """Represents a horizontal alignment axis."""
    
    def __init__(self, y: float, seed_word: Word, axis_id: int):
        self.y = y
        self.seed_word = seed_word
        self.id = axis_id
        self.words: List[Word] = [seed_word]
    
    def add_word(self, word: Word):
        if word not in self.words and not word.merged:
            self.words.append(word)
            word.axis_id = self.id
    
    def remove_word(self, word: Word):
        if word in self.words:
            self.words.remove(word)
    
    def get_active_words(self) -> List[Word]:
        """Get non-merged words sorted left to right."""
        return sorted(
            [w for w in self.words if not w.merged],
            key=lambda w: w.x0
        )


def prepare_words(ocr_results: List[Dict]) -> List[Word]:
    """Convert OCR results to Word objects."""
    words = []
    for idx, item in enumerate(ocr_results):
        text = item.get('text', '').strip()
        bbox = item.get('bbox', [])
        if not text or len(bbox) < 4:
            continue
        words.append(Word(text, bbox, idx))
    return words


def step1_select_seed(words: List[Word]) -> Optional[Word]:
    """Step 1: Select next unprocessed word as seed."""
    unprocessed = [w for w in words if not w.aligned and not w.merged]
    if not unprocessed:
        return None
    unprocessed.sort(key=lambda w: (w.cy, w.cx))
    return unprocessed[0]


def step2_create_axis(seed: Word, axis_counter: int) -> Axis:
    """Step 2: Create horizontal axis through seed's center."""
    axis = Axis(seed.cy, seed, axis_counter)
    seed.aligned = True
    seed.axis_id = axis.id
    return axis


def step3_find_collisions(axis: Axis, words: List[Word]) -> List[Word]:
    """Step 3: Find words intersecting the axis."""
    collisions = []
    for word in words:
        if word.aligned or word.merged:
            continue
        if word.intersects_axis(axis.y):
            collisions.append(word)
    collisions.sort(key=lambda w: w.x0)
    return collisions


def step4_realign(word: Word, axis: Axis):
    """Step 4: Move word vertically to align with axis."""
    word.move_to_axis(axis.y)
    word.aligned = True
    axis.add_word(word)


def step5_detect_strong_collisions(axis: Axis) -> List[Tuple[Word, Word]]:
    """Step 5: Detect pairs with strong bounding box overlap."""
    collisions = []
    words = axis.get_active_words()
    
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            w1, w2 = words[i], words[j]
            overlap = w1.bbox_overlap_ratio(w2)
            
            if overlap >= STRONG_COLLISION_THRESHOLD:
                collisions.append((w1, w2))
    
    return collisions


def step6_merge_collisions(axis: Axis, words: List[Word], 
                           collision_pairs: List[Tuple[Word, Word]]) -> List[Word]:
    """Step 6: Merge strongly colliding word pairs."""
    merged_words = []
    
    for w1, w2 in collision_pairs:
        if w1.merged or w2.merged:
            continue
        
        # Merge w2 into w1
        merged = w1.merge_with(w2)
        
        # Update w1's properties
        w1.text = merged.text
        w1.x0, w1.y0, w1.x1, w1.y1 = merged.x0, merged.y0, merged.x1, merged.y1
        
        # Mark w2 as merged (to be removed)
        w2.merged = True
        axis.remove_word(w2)
        
        merged_words.append(w1)
    
    return merged_words


def step7_propagate(axis: Axis, words: List[Word], all_axes: List[Axis],
                    axis_counter: int, processed: Set[int]) -> int:
    """Step 7: Recursive axis propagation with merging."""
    
    # Main propagation loop
    max_iterations = 100
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        changes_made = False
        
        # Step 3-4: Find and align collisions
        collisions = step3_find_collisions(axis, words)
        for word in collisions:
            step4_realign(word, axis)
            changes_made = True
        
        # Step 5-6: Detect and merge strong collisions
        strong_pairs = step5_detect_strong_collisions(axis)
        if strong_pairs:
            step6_merge_collisions(axis, words, strong_pairs)
            changes_made = True
        
        if not changes_made:
            break
    
    # Generate recursive axes for aligned words
    for word in axis.get_active_words():
        if word.id in processed:
            continue
        
        processed.add(word.id)
        
        # Check for unaligned words that would intersect new axis
        potential_y = word.cy
        has_intersections = any(
            w.intersects_axis(potential_y)
            for w in words
            if not w.aligned and not w.merged
        )
        
        if has_intersections:
            new_axis = Axis(potential_y, word, axis_counter)
            axis_counter += 1
            all_axes.append(new_axis)
            axis_counter = step7_propagate(
                new_axis, words, all_axes, axis_counter, processed
            )
    
    return axis_counter


def step8_reconstruct_lines(axes: List[Axis], page_width: int) -> List[List[Word]]:
    """Step 8: Group words by axis, apply spacing rules."""
    lines = []
    right_margin = page_width * RIGHT_MARGIN_RATIO
    
    # Sort axes top to bottom
    sorted_axes = sorted(axes, key=lambda a: a.y)
    
    for axis in sorted_axes:
        sorted_words = axis.get_active_words()
        if not sorted_words:
            continue
        
        current_line = []
        
        for word in sorted_words:
            if word.x1 > right_margin and current_line:
                lines.append(current_line)
                current_line = [word]
            else:
                current_line.append(word)
        
        if current_line:
            lines.append(current_line)
    
    return lines


def step9_generate_output(lines: List[List[Word]]) -> str:
    """Step 9: Generate final text output."""
    if not lines:
        return ""
    
    all_heights = [w.height for line in lines for w in line]
    avg_height = np.median(all_heights) if all_heights else 20
    
    small_gap = SMALL_GAP_RATIO * avg_height
    tab_gap = TAB_GAP_RATIO * avg_height
    
    output_lines = []
    
    for line in lines:
        if not line:
            continue
        
        parts = []
        
        for i, word in enumerate(line):
            if i == 0:
                parts.append(word.text)
                continue
            
            prev_word = line[i - 1]
            gap = word.x0 - prev_word.x1
            
            if gap < 0:
                parts.append(" ")
            elif gap < small_gap:
                parts.append(" ")
            elif gap > tab_gap:
                parts.append("   ")
            else:
                parts.append(" ")
            
            parts.append(word.text)
        
        output_lines.append("".join(parts))
    
    return "\n".join(output_lines)


def reconstruct_document_layout(words: List[Dict], page_width: int, page_height: int) -> Tuple[str, Dict]:
    """
    Main entry point: Reconstruct document using axis-based realignment with merging.
    
    Args:
        words: List of OCR word detections with 'bbox' and 'text'
        page_width: Page width in pixels
        page_height: Page height in pixels
    
    Returns:
        Tuple of (reconstructed_text, stats_dict)
        stats_dict contains: input_blocks, output_blocks, merged_count, line_count
    """
    word_objects = prepare_words(words)
    input_block_count = len(word_objects)
    
    if not word_objects:
        return "", {"input_blocks": 0, "output_blocks": 0, "merged_count": 0, "line_count": 0}
    
    axes = []
    axis_counter = 0
    processed: Set[int] = set()
    
    # Main loop
    while True:
        seed = step1_select_seed(word_objects)
        if seed is None:
            break
        
        processed.add(seed.id)
        
        axis = step2_create_axis(seed, axis_counter)
        axis_counter += 1
        axes.append(axis)
        
        axis_counter = step7_propagate(
            axis, word_objects, axes, axis_counter, processed
        )
    
    # Count non-merged blocks
    output_block_count = len([w for w in word_objects if not w.merged])
    merged_count = input_block_count - output_block_count
    
    # Step 8-9
    lines = step8_reconstruct_lines(axes, page_width)
    output = step9_generate_output(lines)
    
    stats = {
        "input_blocks": input_block_count,
        "output_blocks": output_block_count,
        "merged_count": merged_count,
        "line_count": len(lines)
    }
    
    return output, stats

