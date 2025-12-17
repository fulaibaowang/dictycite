from .epmc_xml import get_epmc_text
from .ncbi_bioc import get_ncbi_text
from .my_custom import get_epmc_text_my, get_ncbi_text_my

__all__ = [
    "get_epmc_text",
    "get_ncbi_text",
    "get_epmc_text_my",
    "get_ncbi_text_my",
]
