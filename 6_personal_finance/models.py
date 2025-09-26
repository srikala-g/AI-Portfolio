"""
Data models for the Personal Finance Expense Categorizer.
Defines the core data structures for transactions, categories, and analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple
import json


class TransactionType(Enum):
    """Enumeration for transaction types."""
    DEBIT = "debit"
    CREDIT = "credit"


class ExpenseCategory(Enum):
    """Enumeration for expense categories."""
    GROCERIES = "groceries"
    RENT = "rent"
    UTILITIES = "utilities"
    TRANSPORTATION = "transportation"
    DINING = "dining"
    ENTERTAINMENT = "entertainment"
    HEALTHCARE = "healthcare"
    SHOPPING = "shopping"
    TRAVEL = "travel"
    EDUCATION = "education"
    INSURANCE = "insurance"
    SAVINGS = "savings"
    INVESTMENTS = "investments"
    OTHER = "other"


@dataclass
class Transaction:
    """Represents a financial transaction."""
    id: str
    description: str
    amount: float
    date: datetime
    transaction_type: TransactionType
    category: Optional[ExpenseCategory] = None
    confidence_score: Optional[float] = None
    merchant: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert transaction to dictionary for serialization."""
        return {
            'id': self.id,
            'description': self.description,
            'amount': self.amount,
            'date': self.date.isoformat(),
            'transaction_type': self.transaction_type.value,
            'category': self.category.value if self.category else None,
            'confidence_score': self.confidence_score,
            'merchant': self.merchant
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Transaction':
        """Create transaction from dictionary."""
        return cls(
            id=data['id'],
            description=data['description'],
            amount=data['amount'],
            date=datetime.fromisoformat(data['date']),
            transaction_type=TransactionType(data['transaction_type']),
            category=ExpenseCategory(data['category']) if data.get('category') else None,
            confidence_score=data.get('confidence_score'),
            merchant=data.get('merchant')
        )


@dataclass
class CategoryRule:
    """Represents a rule for categorizing transactions."""
    keywords: List[str]
    category: ExpenseCategory
    confidence: float = 1.0
    description: str = ""
    
    def matches(self, transaction_description: str) -> bool:
        """Check if this rule matches the transaction description."""
        description_lower = transaction_description.lower()
        return any(keyword.lower() in description_lower for keyword in self.keywords)


@dataclass
class ExpenseAnalysis:
    """Represents the analysis results for a set of transactions."""
    total_expenses: float
    total_income: float
    net_balance: float
    category_breakdown: Dict[ExpenseCategory, float] = field(default_factory=dict)
    category_counts: Dict[ExpenseCategory, int] = field(default_factory=dict)
    top_categories: List[Tuple[ExpenseCategory, float]] = field(default_factory=list)
    uncategorized_count: int = 0
    analysis_date: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert analysis to dictionary for serialization."""
        return {
            'total_expenses': self.total_expenses,
            'total_income': self.total_income,
            'net_balance': self.net_balance,
            'category_breakdown': {cat.value: amount for cat, amount in self.category_breakdown.items()},
            'category_counts': {cat.value: count for cat, count in self.category_counts.items()},
            'top_categories': [(cat.value, amount) for cat, amount in self.top_categories],
            'uncategorized_count': self.uncategorized_count,
            'analysis_date': self.analysis_date.isoformat()
        }


@dataclass
class CategorizationResult:
    """Represents the result of categorizing a transaction."""
    transaction: Transaction
    predicted_category: ExpenseCategory
    confidence_score: float
    matched_rules: List[CategoryRule] = field(default_factory=list)
    reasoning: str = ""
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary for serialization."""
        return {
            'transaction': self.transaction.to_dict(),
            'predicted_category': self.predicted_category.value,
            'confidence_score': self.confidence_score,
            'matched_rules': [
                {
                    'keywords': rule.keywords,
                    'category': rule.category.value,
                    'confidence': rule.confidence,
                    'description': rule.description
                }
                for rule in self.matched_rules
            ],
            'reasoning': self.reasoning
        }
