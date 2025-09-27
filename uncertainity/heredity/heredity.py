"""
Heredity Probability Calculator

This program calculates the probability of gene inheritance and trait expression
in a family tree using Bayesian inference. It models genetic inheritance patterns
considering gene mutations and trait probabilities.

Key Features:
- Calculates gene inheritance probabilities (0, 1, or 2 copies of a gene)
- Models trait expression based on gene count
- Handles family relationships and parental gene inheritance
- Accounts for gene mutations during inheritance
- Uses Bayesian inference to compute joint probabilities

Usage:
    python heredity.py <data.csv>

Input CSV Format:
    The CSV file should contain columns: name, mother, father, trait
    - name: Person's name (string)
    - mother: Mother's name (string, or blank if unknown)
    - father: Father's name (string, or blank if unknown)  
    - trait: 1 if person has trait, 0 if person doesn't have trait, blank if unknown

Example CSV:
    name,mother,father,trait
    Harry,,,1
    James,Harry,,0
    Lily,Harry,,1

Output:
    For each person, displays:
    - Gene probabilities: P(0 genes), P(1 gene), P(2 genes)
    - Trait probabilities: P(has trait), P(no trait)

Algorithm:
    1. Loads family data from CSV
    2. Generates all possible gene and trait combinations
    3. Calculates joint probabilities using inheritance rules
    4. Normalizes probabilities to sum to 1
    5. Outputs probability distributions for each person

Genetic Model:
    - Gene inheritance follows Mendelian rules with 1% mutation rate
    - Trait expression depends on gene count (0, 1, or 2 copies)
    - Unconditional gene probabilities: 96% (0 genes), 3% (1 gene), 1% (2 genes)
    - Trait probabilities given gene count: 2 genes (65% trait), 1 gene (56% trait), 0 genes (1% trait)
"""

import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    prob = 1

    for person in people:
        # Determine gene count for this person
        if person in two_genes:
            genes = 2
        elif person in one_gene:
            genes = 1
        else:
            genes = 0

        # Trait status for this person
        has_trait = person in have_trait

        mother = people[person]["mother"]
        father = people[person]["father"]

        # If no parental info, use unconditional probability
        if mother is None and father is None:
            gene_prob = PROBS["gene"][genes]
        else:
            # Probabilities parent passes gene
            def pass_prob(parent):
                if parent in two_genes:
                    return 1 - PROBS["mutation"]
                elif parent in one_gene:
                    return 0.5
                else:
                    return PROBS["mutation"]

            mom_prob = pass_prob(mother)
            dad_prob = pass_prob(father)

            if genes == 2:
                gene_prob = mom_prob * dad_prob
            elif genes == 1:
                gene_prob = mom_prob * (1 - dad_prob) + (1 - mom_prob) * dad_prob
            else:  # genes == 0
                gene_prob = (1 - mom_prob) * (1 - dad_prob)

        trait_prob = PROBS["trait"][genes][has_trait]
        prob *= gene_prob * trait_prob

    return prob


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:
        # Determine gene count for this person
        if person in two_genes:
            genes = 2
        elif person in one_gene:
            genes = 1
        else:
            genes = 0

        # Trait status for this person
        has_trait = person in have_trait

        # Update gene probability
        probabilities[person]["gene"][genes] += p
        # Update trait probability
        probabilities[person]["trait"][has_trait] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:
        # Normalize gene probabilities
        gene_total = sum(probabilities[person]["gene"].values())
        if gene_total != 0:
            for gene in probabilities[person]["gene"]:
                probabilities[person]["gene"][gene] /= gene_total

        # Normalize trait probabilities
        trait_total = sum(probabilities[person]["trait"].values())
        if trait_total != 0:
            for trait in probabilities[person]["trait"]:
                probabilities[person]["trait"][trait] /= trait_total


if __name__ == "__main__":
    main()
