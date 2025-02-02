import omf_python


def is_omf(filepath: str) -> bool:
    r"""
    Returns `True` if the file appears to be a valid OMF file.
    """
    if omf_python.detect_omf1(filepath):
        return True

    try:
        # Attempt to read as an OMF2 file. If project can be read consider it a valid OMF2 file.
        reader = omf_python.Reader(filepath)
        reader.project()
        return True
    except omf_python.OmfException:
        pass

    return False
