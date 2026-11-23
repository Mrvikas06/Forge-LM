"""Text cleaning and normalization functions for dataset preprocessing."""


def clean_text(text: str, min_length: int = 20) -> str:
    """Clean and normalize raw text.

    - Strips leading/trailing whitespace
    - Normalizes consecutive whitespace into single spaces
    - Rejects invalid types or strings shorter than min_length

    Args:
        text: Raw input text string.
        min_length: Minimum character length to retain.

    Returns:
        Cleaned string, or empty string if input is invalid/too short.
    """
    if not text or not isinstance(text, str):
        return ""

    # Normalize whitespace
    cleaned = " ".join(text.split()).strip()

    if len(cleaned) < min_length:
        return ""

    return cleaned
