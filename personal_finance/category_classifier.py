"""
Text classification service for categorizing financial transactions.
Uses keyword-based rules and machine learning for transaction categorization.
"""

import re
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from models import (
    Transaction, ExpenseCategory, CategoryRule, 
    CategorizationResult, TransactionType
)


class CategoryClassifier:
    """Service for categorizing financial transactions using text classification."""
    
    def __init__(self):
        """Initialize the classifier with default category rules."""
        self.rules = self._initialize_default_rules()
        self.merchant_patterns = self._initialize_merchant_patterns()
    
    def _initialize_default_rules(self) -> List[CategoryRule]:
        """Initialize default categorization rules."""
        return [
            # Groceries
            CategoryRule(
                keywords=["grocery", "supermarket", "walmart", "target", "kroger", "safeway", "whole foods", "trader joe", "food", "market"],
                category=ExpenseCategory.GROCERIES,
                confidence=0.9,
                description="Grocery store purchases"
            ),
            
            # Rent/Housing
            CategoryRule(
                keywords=["rent", "apartment", "housing", "mortgage", "lease", "landlord"],
                category=ExpenseCategory.RENT,
                confidence=0.95,
                description="Housing expenses"
            ),
            
            # Utilities
            CategoryRule(
                keywords=["electric", "gas", "water", "sewer", "trash", "utility", "power", "internet", "cable", "phone"],
                category=ExpenseCategory.UTILITIES,
                confidence=0.9,
                description="Utility bills"
            ),
            
            # Transportation
            CategoryRule(
                keywords=["gas", "fuel", "gas station", "uber", "lyft", "taxi", "bus", "train", "metro", "parking", "toll", "car", "auto"],
                category=ExpenseCategory.TRANSPORTATION,
                confidence=0.85,
                description="Transportation expenses"
            ),
            
            # Dining
            CategoryRule(
                keywords=["restaurant", "cafe", "coffee", "starbucks", "mcdonald", "burger", "pizza", "dining", "food", "lunch", "dinner", "breakfast"],
                category=ExpenseCategory.DINING,
                confidence=0.8,
                description="Dining out"
            ),
            
            # Entertainment
            CategoryRule(
                keywords=["movie", "cinema", "theater", "netflix", "spotify", "amazon prime", "gym", "fitness", "sports", "game", "entertainment"],
                category=ExpenseCategory.ENTERTAINMENT,
                confidence=0.8,
                description="Entertainment expenses"
            ),
            
            # Healthcare
            CategoryRule(
                keywords=["doctor", "hospital", "pharmacy", "medical", "health", "dental", "vision", "insurance", "clinic"],
                category=ExpenseCategory.HEALTHCARE,
                confidence=0.9,
                description="Healthcare expenses"
            ),
            
            # Shopping
            CategoryRule(
                keywords=["amazon", "ebay", "shop", "store", "mall", "clothing", "shoes", "electronics", "retail"],
                category=ExpenseCategory.SHOPPING,
                confidence=0.7,
                description="General shopping"
            ),
            
            # Travel
            CategoryRule(
                keywords=["hotel", "flight", "airline", "travel", "vacation", "trip", "booking", "airbnb", "expedia"],
                category=ExpenseCategory.TRAVEL,
                confidence=0.9,
                description="Travel expenses"
            ),
            
            # Education
            CategoryRule(
                keywords=["school", "university", "college", "tuition", "education", "course", "book", "textbook", "student"],
                category=ExpenseCategory.EDUCATION,
                confidence=0.9,
                description="Education expenses"
            ),
            
            # Insurance
            CategoryRule(
                keywords=["insurance", "premium", "coverage", "policy"],
                category=ExpenseCategory.INSURANCE,
                confidence=0.9,
                description="Insurance payments"
            ),
            
            # Savings/Investments
            CategoryRule(
                keywords=["savings", "investment", "401k", "ira", "retirement", "portfolio", "stocks", "bonds"],
                category=ExpenseCategory.SAVINGS,
                confidence=0.8,
                description="Savings and investments"
            )
        ]
    
    def _initialize_merchant_patterns(self) -> Dict[str, ExpenseCategory]:
        """Initialize merchant-specific patterns."""
        return {
            r".*walmart.*": ExpenseCategory.GROCERIES,
            r".*target.*": ExpenseCategory.GROCERIES,
            r".*kroger.*": ExpenseCategory.GROCERIES,
            r".*safeway.*": ExpenseCategory.GROCERIES,
            r".*whole.foods.*": ExpenseCategory.GROCERIES,
            r".*trader.joe.*": ExpenseCategory.GROCERIES,
            r".*starbucks.*": ExpenseCategory.DINING,
            r".*mcdonald.*": ExpenseCategory.DINING,
            r".*uber.*": ExpenseCategory.TRANSPORTATION,
            r".*lyft.*": ExpenseCategory.TRANSPORTATION,
            r".*amazon.*": ExpenseCategory.SHOPPING,
            r".*netflix.*": ExpenseCategory.ENTERTAINMENT,
            r".*spotify.*": ExpenseCategory.ENTERTAINMENT,
            r".*gas.station.*": ExpenseCategory.TRANSPORTATION,
            r".*shell.*": ExpenseCategory.TRANSPORTATION,
            r".*exxon.*": ExpenseCategory.TRANSPORTATION,
            r".*chevron.*": ExpenseCategory.TRANSPORTATION,
        }
    
    def categorize_transaction(self, transaction: Transaction) -> CategorizationResult:
        """
        Categorize a single transaction.
        
        Args:
            transaction: The transaction to categorize
            
        Returns:
            CategorizationResult with predicted category and confidence
        """
        if transaction.transaction_type == TransactionType.CREDIT:
            # Credits are typically income, categorize as savings
            return CategorizationResult(
                transaction=transaction,
                predicted_category=ExpenseCategory.SAVINGS,
                confidence_score=0.9,
                reasoning="Credit transaction categorized as savings/income"
            )
        
        # Check merchant patterns first
        merchant_category = self._check_merchant_patterns(transaction.description)
        if merchant_category:
            return CategorizationResult(
                transaction=transaction,
                predicted_category=merchant_category,
                confidence_score=0.9,
                reasoning=f"Matched merchant pattern for {merchant_category.value}"
            )
        
        # Check keyword rules
        matched_rules = []
        for rule in self.rules:
            if rule.matches(transaction.description):
                matched_rules.append(rule)
        
        if matched_rules:
            # Use the rule with highest confidence
            best_rule = max(matched_rules, key=lambda r: r.confidence)
            return CategorizationResult(
                transaction=transaction,
                predicted_category=best_rule.category,
                confidence_score=best_rule.confidence,
                matched_rules=matched_rules,
                reasoning=f"Matched {len(matched_rules)} rules, best match: {best_rule.description}"
            )
        
        # Default to "Other" if no rules match
        return CategorizationResult(
            transaction=transaction,
            predicted_category=ExpenseCategory.OTHER,
            confidence_score=0.3,
            reasoning="No matching rules found, categorized as Other"
        )
    
    def _check_merchant_patterns(self, description: str) -> Optional[ExpenseCategory]:
        """Check if description matches any merchant patterns."""
        description_lower = description.lower()
        for pattern, category in self.merchant_patterns.items():
            if re.search(pattern, description_lower):
                return category
        return None
    
    def categorize_batch(self, transactions: List[Transaction]) -> List[CategorizationResult]:
        """
        Categorize multiple transactions.
        
        Args:
            transactions: List of transactions to categorize
            
        Returns:
            List of CategorizationResult objects
        """
        return [self.categorize_transaction(transaction) for transaction in transactions]
    
    def add_custom_rule(self, rule: CategoryRule) -> None:
        """Add a custom categorization rule."""
        self.rules.append(rule)
    
    def get_category_statistics(self, results: List[CategorizationResult]) -> Dict[ExpenseCategory, Dict]:
        """Get statistics for categorized transactions."""
        stats = defaultdict(lambda: {'count': 0, 'total_amount': 0.0, 'avg_confidence': 0.0})
        
        for result in results:
            category = result.predicted_category
            stats[category]['count'] += 1
            stats[category]['total_amount'] += result.transaction.amount
            stats[category]['avg_confidence'] += result.confidence_score
        
        # Calculate averages
        for category_data in stats.values():
            if category_data['count'] > 0:
                category_data['avg_confidence'] /= category_data['count']
        
        return dict(stats)
