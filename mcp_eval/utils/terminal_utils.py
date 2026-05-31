"""Terminal display utilities for proper text alignment.

This module provides utilities for calculating visible width of text in terminals,
handling ANSI escape codes, emoji, and East Asian characters correctly.
"""

import re
import unicodedata

# Compile regex once at module level for performance
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")

# Unicode ranges for emoji
EMOJI_START = 0x1F300
EMOJI_END = 0x1FAFF


def calculate_display_width(text: str) -> int:
    """Calculate the visible width of text in terminal columns.

    This function correctly handles:
    - ANSI escape codes (removed from width calculation)
    - Emoji characters (counted as 2 columns)
    - East Asian Wide/Fullwidth characters (counted as 2 columns)
    - Combining characters (counted as 0 columns)
    - Regular ASCII characters (counted as 1 column)

    Args:
        text: Input text that may contain ANSI codes, emoji, or unicode characters

    Returns:
        Number of terminal columns the text will occupy when displayed

    Examples:
        >>> calculate_display_width("Hello")
        5
        >>> calculate_display_width("ABCD")
        4
        >>> calculate_display_width("🤖")
        2
        >>> calculate_display_width("\033[31mRed\033[0m")
        3
    """
    # Remove ANSI escape codes (they don't occupy display space)
    clean_text = ANSI_ESCAPE_RE.sub("", text)

    width = 0
    for char in clean_text:
        # Skip combining characters (zero width)
        if unicodedata.combining(char):
            continue

        code_point = ord(char)

        # Emoji range (most common emoji, counted as 2 columns)
        if EMOJI_START <= code_point <= EMOJI_END:
            width += 2
            continue

        # East Asian Width property
        # W = Wide, F = Fullwidth (both occupy 2 columns)
        eaw = unicodedata.east_asian_width(char)
        if eaw in ("W", "F"):
            width += 2
        else:
            width += 1

    return width


def truncate_with_ellipsis(text: str, max_width: int, ellipsis: str = "…") -> str:
    """Truncate text to fit within max_width, adding ellipsis if needed.

    Args:
        text: Text to truncate (ANSI codes are preserved but not counted)
        max_width: Maximum visible width in terminal columns
        ellipsis: Ellipsis character to use (default: "…")

    Returns:
        Truncated text with ellipsis if needed

    Examples:
        >>> truncate_with_ellipsis("Hello World", 8)
        'Hello W…'
        >>> truncate_with_ellipsis("ABCDEFGH", 5)
        'ABCD…'
    """
    if max_width <= 0:
        return ""

    current_width = calculate_display_width(text)

    # No truncation needed
    if current_width <= max_width:
        return text

    # Remove ANSI codes for truncation (we'll lose color, but that's expected)
    plain_text = ANSI_ESCAPE_RE.sub("", text)

    # If max_width is too small for ellipsis
    ellipsis_width = calculate_display_width(ellipsis)
    if max_width <= ellipsis_width:
        return plain_text[:max_width]

    # Find truncation point
    available_width = max_width - ellipsis_width
    truncated = ""
    current_width = 0

    for char in plain_text:
        char_width = calculate_display_width(char)
        if current_width + char_width > available_width:
            break
        truncated += char
        current_width += char_width

    return truncated + ellipsis


def pad_to_width(text: str, target_width: int, align: str = "left", fill_char: str = " ") -> str:
    """Pad text to reach target width with proper alignment.

    Args:
        text: Text to pad (may contain ANSI codes)
        target_width: Target width in terminal columns
        align: Alignment mode - "left", "right", or "center"
        fill_char: Character to use for padding (default: space)

    Returns:
        Padded text

    Examples:
        >>> pad_to_width("Hello", 10)
        'Hello     '
        >>> pad_to_width("Test", 10)
        'Test      '
        >>> pad_to_width("Test", 10, align="center")
        '   Test   '
    """
    current_width = calculate_display_width(text)

    if current_width >= target_width:
        return text

    padding_needed = target_width - current_width

    if align == "left":
        return text + (fill_char * padding_needed)
    elif align == "right":
        return (fill_char * padding_needed) + text
    elif align == "center":
        left_padding = padding_needed // 2
        right_padding = padding_needed - left_padding
        return (fill_char * left_padding) + text + (fill_char * right_padding)
    else:
        raise ValueError(f"Invalid align value: {align}. Must be 'left', 'right', or 'center'")
