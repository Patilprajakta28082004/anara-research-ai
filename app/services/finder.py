"""
Literature search: queries arXiv and Semantic Scholar concurrently and
merges the results into one normalised shape the frontend can render.
"""
import asyncio
import xml.etree.ElementTree as ET
from typing import Any, Dict, List

import httpx

ARXIV_API = "http://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"


def _text_or_default(el, default: str = "") -> str:
    """ElementTree nodes can legitimately be missing; never let that crash a request."""
    if el is None or el.text is None:
        return default
    return el.text.replace("\n", " ").strip()


async def search_arxiv(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search the arXiv API using a keyword query."""
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
    }

    results: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(follow_redirects=True) as client:
        try:
            response = await client.get(ARXIV_API, params=params, timeout=10.0)
            if response.status_code != 200:
                print(f"ArXiv failed with code {response.status_code}")
                return results

            root = ET.fromstring(response.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            for entry in root.findall("atom:entry", ns):
                title = _text_or_default(entry.find("atom:title", ns), "Untitled")
                summary = _text_or_default(entry.find("atom:summary", ns), "")
                published_el = entry.find("atom:published", ns)
                year = published_el.text[:4] if published_el is not None and published_el.text else "Unknown"
                authors = [
                    _text_or_default(author.find("atom:name", ns))
                    for author in entry.findall("atom:author", ns)
                ]
                id_el = entry.find("atom:id", ns)
                link = id_el.text.strip() if id_el is not None and id_el.text else ""
                pdf_link = link.replace("abs", "pdf") + ".pdf" if link else None

                results.append(
                    {
                        "id": link,
                        "title": title,
                        "authors": authors,
                        "year": year,
                        "abstract": summary or "No abstract available.",
                        "url": link,
                        "pdf_url": pdf_link,
                        "source": "arXiv",
                    }
                )
        except Exception as e:
            print(f"Error fetching from arXiv: {e}")

    return results


async def search_semantic_scholar(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search Semantic Scholar API."""
    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,authors,year,abstract,url,openAccessPdf",
    }

    results: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(follow_redirects=True) as client:
        try:
            # Semantic Scholar rate-limits unauthenticated traffic; failures are
            # expected sometimes and should not break the overall search.
            response = await client.get(SEMANTIC_SCHOLAR_API, params=params, timeout=10.0)
            if response.status_code != 200:
                print(f"Semantic Scholar failed with code {response.status_code}")
                return results

            data = response.json()
            for item in data.get("data", []):
                author_list = item.get("authors") or []
                authors = [a.get("name", "") for a in author_list]
                pdf_link = (item.get("openAccessPdf") or {}).get("url")

                results.append(
                    {
                        "id": item.get("paperId"),
                        "title": item.get("title") or "Untitled",
                        "authors": authors,
                        "year": str(item.get("year")) if item.get("year") else "Unknown",
                        "abstract": item.get("abstract") or "No abstract available.",
                        "url": item.get("url"),
                        "pdf_url": pdf_link,
                        "source": "Semantic Scholar",
                    }
                )
        except Exception as e:
            print(f"Error fetching from Semantic Scholar: {e}")

    return results


async def search_literature(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Aggregates literature search from multiple sources, run concurrently."""
    per_source = max(1, limit // 2)

    arxiv_res, ss_res = await asyncio.gather(
        search_arxiv(query, max_results=per_source),
        search_semantic_scholar(query, max_results=per_source),
        return_exceptions=True,
    )

    results: List[Dict[str, Any]] = []
    if not isinstance(arxiv_res, Exception):
        results.extend(arxiv_res)
    if not isinstance(ss_res, Exception):
        results.extend(ss_res)

    return results
