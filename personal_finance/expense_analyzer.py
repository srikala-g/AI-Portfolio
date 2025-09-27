"""
Expense analyzer for financial transaction analysis and reporting.
Provides insights into spending patterns and category breakdowns.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from models import (
    Transaction, ExpenseCategory, ExpenseAnalysis, 
    CategorizationResult, TransactionType
)
from category_classifier import CategoryClassifier


class ExpenseAnalyzer:
    """Analyzer for financial transactions with reporting capabilities."""
    
    def __init__(self, classifier: Optional[CategoryClassifier] = None):
        """
        Initialize the expense analyzer.
        
        Args:
            classifier: Optional custom classifier, uses default if None
        """
        self.classifier = classifier or CategoryClassifier()
        self.transactions: List[Transaction] = []
        self.categorized_transactions: List[CategorizationResult] = []
    
    def add_transaction(self, transaction: Transaction) -> CategorizationResult:
        """
        Add a single transaction and categorize it.
        
        Args:
            transaction: Transaction to add and categorize
            
        Returns:
            CategorizationResult for the transaction
        """
        self.transactions.append(transaction)
        result = self.classifier.categorize_transaction(transaction)
        self.categorized_transactions.append(result)
        return result
    
    def add_transactions(self, transactions: List[Transaction]) -> List[CategorizationResult]:
        """
        Add multiple transactions and categorize them.
        
        Args:
            transactions: List of transactions to add and categorize
            
        Returns:
            List of CategorizationResult objects
        """
        self.transactions.extend(transactions)
        results = self.classifier.categorize_batch(transactions)
        self.categorized_transactions.extend(results)
        return results
    
    def analyze_expenses(self, 
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> ExpenseAnalysis:
        """
        Analyze expenses for a given date range.
        
        Args:
            start_date: Start date for analysis (inclusive)
            end_date: End date for analysis (inclusive)
            
        Returns:
            ExpenseAnalysis object with analysis results
        """
        # Filter transactions by date range
        filtered_transactions = self._filter_transactions_by_date(start_date, end_date)
        filtered_results = self._filter_results_by_date(start_date, end_date)
        
        # Calculate totals
        total_expenses = sum(
            result.transaction.amount 
            for result in filtered_results 
            if result.transaction.transaction_type == TransactionType.DEBIT
        )
        
        total_income = sum(
            result.transaction.amount 
            for result in filtered_results 
            if result.transaction.transaction_type == TransactionType.CREDIT
        )
        
        net_balance = total_income - total_expenses
        
        # Category breakdown
        category_breakdown = defaultdict(float)
        category_counts = defaultdict(int)
        
        for result in filtered_results:
            if result.transaction.transaction_type == TransactionType.DEBIT:
                category_breakdown[result.predicted_category] += result.transaction.amount
                category_counts[result.predicted_category] += 1
        
        # Top categories by amount
        top_categories = sorted(
            category_breakdown.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Count uncategorized transactions
        uncategorized_count = sum(
            1 for result in filtered_results 
            if result.predicted_category == ExpenseCategory.OTHER
        )
        
        return ExpenseAnalysis(
            total_expenses=total_expenses,
            total_income=total_income,
            net_balance=net_balance,
            category_breakdown=dict(category_breakdown),
            category_counts=dict(category_counts),
            top_categories=top_categories,
            uncategorized_count=uncategorized_count,
            analysis_date=datetime.now()
        )
    
    def get_spending_trends(self, days: int = 30) -> Dict[str, List[Tuple[datetime, float]]]:
        """
        Get spending trends over time.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with category trends
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Group transactions by date and category
        daily_spending = defaultdict(lambda: defaultdict(float))
        
        for result in self.categorized_transactions:
            transaction = result.transaction
            if (start_date <= transaction.date <= end_date and 
                transaction.transaction_type == TransactionType.DEBIT):
                
                date_key = transaction.date.date()
                daily_spending[result.predicted_category][date_key] += transaction.amount
        
        # Convert to sorted lists
        trends = {}
        for category, daily_amounts in daily_spending.items():
            sorted_dates = sorted(daily_amounts.items())
            trends[category.value] = [(datetime.combine(date, datetime.min.time()), amount) 
                                   for date, amount in sorted_dates]
        
        return trends
    
    def get_category_insights(self, category: ExpenseCategory) -> Dict:
        """
        Get detailed insights for a specific category.
        
        Args:
            category: Category to analyze
            
        Returns:
            Dictionary with category insights
        """
        category_transactions = [
            result for result in self.categorized_transactions
            if result.predicted_category == category
        ]
        
        if not category_transactions:
            return {
                'category': category.value,
                'total_amount': 0,
                'transaction_count': 0,
                'average_amount': 0,
                'max_transaction': 0,
                'min_transaction': 0,
                'confidence_avg': 0
            }
        
        amounts = [result.transaction.amount for result in category_transactions]
        confidences = [result.confidence_score for result in category_transactions]
        
        return {
            'category': category.value,
            'total_amount': sum(amounts),
            'transaction_count': len(amounts),
            'average_amount': sum(amounts) / len(amounts),
            'max_transaction': max(amounts),
            'min_transaction': min(amounts),
            'confidence_avg': sum(confidences) / len(confidences),
            'recent_transactions': [
                {
                    'description': result.transaction.description,
                    'amount': result.transaction.amount,
                    'date': result.transaction.date.isoformat(),
                    'confidence': result.confidence_score
                }
                for result in sorted(category_transactions, 
                                  key=lambda x: x.transaction.date, 
                                  reverse=True)[:5]
            ]
        }
    
    def export_analysis(self, format_type: str = "json") -> str:
        """
        Export analysis results in specified format.
        
        Args:
            format_type: Export format ("json", "csv")
            
        Returns:
            Exported data as string
        """
        if format_type == "json":
            return self._export_json()
        elif format_type == "csv":
            return self._export_csv()
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _filter_transactions_by_date(self, 
                                   start_date: Optional[datetime], 
                                   end_date: Optional[datetime]) -> List[Transaction]:
        """Filter transactions by date range."""
        filtered = self.transactions
        
        if start_date:
            filtered = [t for t in filtered if t.date >= start_date]
        if end_date:
            filtered = [t for t in filtered if t.date <= end_date]
        
        return filtered
    
    def _filter_results_by_date(self, 
                              start_date: Optional[datetime], 
                              end_date: Optional[datetime]) -> List[CategorizationResult]:
        """Filter categorization results by date range."""
        filtered = self.categorized_transactions
        
        if start_date:
            filtered = [r for r in filtered if r.transaction.date >= start_date]
        if end_date:
            filtered = [r for r in filtered if r.transaction.date <= end_date]
        
        return filtered
    
    def _export_json(self) -> str:
        """Export data as JSON."""
        import json
        
        data = {
            'transactions': [result.to_dict() for result in self.categorized_transactions],
            'analysis': self.analyze_expenses().to_dict()
        }
        
        return json.dumps(data, indent=2, default=str)
    
    def _export_csv(self) -> str:
        """Export data as CSV."""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            'Date', 'Description', 'Amount', 'Type', 'Category', 
            'Confidence', 'Merchant'
        ])
        
        # Write transactions
        for result in self.categorized_transactions:
            transaction = result.transaction
            writer.writerow([
                transaction.date.strftime('%Y-%m-%d'),
                transaction.description,
                transaction.amount,
                transaction.transaction_type.value,
                result.predicted_category.value,
                result.confidence_score,
                transaction.merchant or ''
            ])
        
        return output.getvalue()
    
    def clear_data(self) -> None:
        """Clear all stored transactions and results."""
        self.transactions.clear()
        self.categorized_transactions.clear()
