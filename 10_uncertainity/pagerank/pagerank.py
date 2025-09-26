"""
PageRank Algorithm Implementation

This module implements the PageRank algorithm using two different approaches:
1. Sampling method: Simulates random web surfing to estimate PageRank values
2. Iterative method: Uses matrix calculations to converge to exact PageRank values

The algorithm analyzes a corpus of HTML pages and calculates their relative importance
based on link structure, similar to how Google's original PageRank algorithm works.

USAGE:
    python pagerank.py <directory_path>
    
    Where <directory_path> is a directory containing HTML files with links between them.

EXAMPLE:
    python pagerank.py ./corpus
    
    This will analyze all HTML files in the ./corpus directory and calculate PageRank
    values for each page based on the links between them.

ALGORITHM DETAILS:
    - Damping factor: 0.85 (probability of following links vs. random jump)
    - Sampling: 10,000 random walks to estimate PageRank
    - Iteration: Converges when changes are < 0.001
    - Handles pages with no outgoing links (dangling nodes)

OUTPUT:
    The program outputs PageRank values for each page using both methods,
    allowing comparison between sampling and iterative approaches.
"""

import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    distribution = dict()
    pages = list(corpus.keys())
    n_pages = len(pages)
    links = corpus[page]

    if links:
        for p in pages:
            distribution[p] = (1 - damping_factor) / n_pages
            if p in links:
                distribution[p] += damping_factor / len(links)
    else:
        for p in pages:
            distribution[p] = 1 / n_pages

    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    counts = {page: 0 for page in corpus}
    pages = list(corpus.keys())

    # First sample: random page
    current_page = random.choice(pages)
    counts[current_page] += 1

    for _ in range(1, n):
        probs = transition_model(corpus, current_page, damping_factor)
        next_page = random.choices(
            population=list(probs.keys()),
            weights=list(probs.values()),
            k=1
        )[0]
        counts[next_page] += 1
        current_page = next_page

    # Normalize counts to get probabilities
    pagerank = {page: counts[page] / n for page in corpus}
    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = len(corpus)
    pagerank = {page: 1 / N for page in corpus}
    converged = False

    while not converged:
        new_pagerank = {}
        for page in corpus:
            total = 0
            for possible_page in corpus:
                links = corpus[possible_page]
                if links:
                    if page in links:
                        total += pagerank[possible_page] / len(links)
                else:
                    total += pagerank[possible_page] / N
            new_pagerank[page] = (1 - damping_factor) / N + damping_factor * total

        converged = all(abs(new_pagerank[page] - pagerank[page]) < 0.001 for page in pagerank)
        pagerank = new_pagerank.copy()

    # Normalize to ensure sum is 1
    total_rank = sum(pagerank.values())
    pagerank = {page: rank / total_rank for page, rank in pagerank.items()}
    return pagerank


if __name__ == "__main__":
    main()
