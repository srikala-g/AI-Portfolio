"""
Comprehensive test suite for the Personal Finance Expense Categorizer.
Tests all components including models, classifier, and analyzer.
"""

import unittest
from datetime import datetime, timedelta
from typing import List

from models import (
    Transaction, TransactionType, ExpenseCategory, CategoryRule,
    CategorizationResult, ExpenseAnalysis
)
from category_classifier import CategoryClassifier
from expense_analyzer import ExpenseAnalyzer


class TestTransactionModel(unittest.TestCase):
    """Test cases for Transaction model."""
    
    def setUp(self):
        """Set up test data."""
        self.transaction = Transaction(
            id="test_001",
            description="WALMART SUPERCENTER",
            amount=85.50,
            date=datetime.now(),
            transaction_type=TransactionType.DEBIT,
            category=ExpenseCategory.GROCERIES,
            confidence_score=0.9,
            merchant="Walmart"
        )
    
    def test_transaction_creation(self):
        """Test transaction creation."""
        self.assertEqual(self.transaction.id, "test_001")
        self.assertEqual(self.transaction.description, "WALMART SUPERCENTER")
        self.assertEqual(self.transaction.amount, 85.50)
        self.assertEqual(self.transaction.transaction_type, TransactionType.DEBIT)
        self.assertEqual(self.transaction.category, ExpenseCategory.GROCERIES)
        self.assertEqual(self.transaction.confidence_score, 0.9)
        self.assertEqual(self.transaction.merchant, "Walmart")
    
    def test_transaction_to_dict(self):
        """Test transaction serialization."""
        data = self.transaction.to_dict()
        self.assertIn('id', data)
        self.assertIn('description', data)
        self.assertIn('amount', data)
        self.assertIn('date', data)
        self.assertIn('transaction_type', data)
        self.assertIn('category', data)
        self.assertIn('confidence_score', data)
        self.assertIn('merchant', data)
    
    def test_transaction_from_dict(self):
        """Test transaction deserialization."""
        data = self.transaction.to_dict()
        new_transaction = Transaction.from_dict(data)
        self.assertEqual(new_transaction.id, self.transaction.id)
        self.assertEqual(new_transaction.description, self.transaction.description)
        self.assertEqual(new_transaction.amount, self.transaction.amount)
        self.assertEqual(new_transaction.transaction_type, self.transaction.transaction_type)
        self.assertEqual(new_transaction.category, self.transaction.category)


class TestCategoryRule(unittest.TestCase):
    """Test cases for CategoryRule model."""
    
    def setUp(self):
        """Set up test data."""
        self.rule = CategoryRule(
            keywords=["walmart", "target", "grocery"],
            category=ExpenseCategory.GROCERIES,
            confidence=0.9,
            description="Grocery store purchases"
        )
    
    def test_rule_creation(self):
        """Test rule creation."""
        self.assertEqual(self.rule.category, ExpenseCategory.GROCERIES)
        self.assertEqual(self.rule.confidence, 0.9)
        self.assertEqual(len(self.rule.keywords), 3)
    
    def test_rule_matches(self):
        """Test rule matching."""
        # Should match
        self.assertTrue(self.rule.matches("WALMART SUPERCENTER"))
        self.assertTrue(self.rule.matches("Target Store"))
        self.assertTrue(self.rule.matches("Grocery shopping"))
        
        # Should not match
        self.assertFalse(self.rule.matches("Starbucks Coffee"))
        self.assertFalse(self.rule.matches("Gas Station"))
    
    def test_case_insensitive_matching(self):
        """Test case insensitive matching."""
        self.assertTrue(self.rule.matches("WALMART"))
        self.assertTrue(self.rule.matches("walmart"))
        self.assertTrue(self.rule.matches("Walmart"))


class TestCategoryClassifier(unittest.TestCase):
    """Test cases for CategoryClassifier."""
    
    def setUp(self):
        """Set up test data."""
        self.classifier = CategoryClassifier()
        self.test_transactions = [
            Transaction("001", "WALMART SUPERCENTER", 85.50, datetime.now(), TransactionType.DEBIT),
            Transaction("002", "STARBUCKS COFFEE", 4.75, datetime.now(), TransactionType.DEBIT),
            Transaction("003", "SHELL GAS STATION", 52.30, datetime.now(), TransactionType.DEBIT),
            Transaction("004", "NETFLIX SUBSCRIPTION", 15.99, datetime.now(), TransactionType.DEBIT),
            Transaction("005", "SALARY DEPOSIT", 3500.00, datetime.now(), TransactionType.CREDIT),
        ]
    
    def test_classifier_initialization(self):
        """Test classifier initialization."""
        self.assertIsInstance(self.classifier, CategoryClassifier)
        self.assertGreater(len(self.classifier.rules), 0)
        self.assertGreater(len(self.classifier.merchant_patterns), 0)
    
    def test_categorize_groceries(self):
        """Test grocery categorization."""
        result = self.classifier.categorize_transaction(self.test_transactions[0])
        self.assertEqual(result.predicted_category, ExpenseCategory.GROCERIES)
        self.assertGreater(result.confidence_score, 0.5)
    
    def test_categorize_dining(self):
        """Test dining categorization."""
        result = self.classifier.categorize_transaction(self.test_transactions[1])
        self.assertEqual(result.predicted_category, ExpenseCategory.DINING)
        self.assertGreater(result.confidence_score, 0.5)
    
    def test_categorize_transportation(self):
        """Test transportation categorization."""
        result = self.classifier.categorize_transaction(self.test_transactions[2])
        self.assertEqual(result.predicted_category, ExpenseCategory.TRANSPORTATION)
        self.assertGreater(result.confidence_score, 0.5)
    
    def test_categorize_entertainment(self):
        """Test entertainment categorization."""
        result = self.classifier.categorize_transaction(self.test_transactions[3])
        self.assertEqual(result.predicted_category, ExpenseCategory.ENTERTAINMENT)
        self.assertGreater(result.confidence_score, 0.5)
    
    def test_categorize_credit_transaction(self):
        """Test credit transaction categorization."""
        result = self.classifier.categorize_transaction(self.test_transactions[4])
        self.assertEqual(result.predicted_category, ExpenseCategory.SAVINGS)
        self.assertGreater(result.confidence_score, 0.5)
    
    def test_categorize_batch(self):
        """Test batch categorization."""
        results = self.classifier.categorize_batch(self.test_transactions)
        self.assertEqual(len(results), len(self.test_transactions))
        
        for result in results:
            self.assertIsInstance(result, CategorizationResult)
            self.assertIsNotNone(result.predicted_category)
            self.assertGreaterEqual(result.confidence_score, 0.0)
            self.assertLessEqual(result.confidence_score, 1.0)
    
    def test_add_custom_rule(self):
        """Test adding custom rules."""
        custom_rule = CategoryRule(
            keywords=["custom", "test"],
            category=ExpenseCategory.OTHER,
            confidence=0.8,
            description="Custom test rule"
        )
        
        initial_rule_count = len(self.classifier.rules)
        self.classifier.add_custom_rule(custom_rule)
        self.assertEqual(len(self.classifier.rules), initial_rule_count + 1)
    
    def test_get_category_statistics(self):
        """Test category statistics."""
        results = self.classifier.categorize_batch(self.test_transactions)
        stats = self.classifier.get_category_statistics(results)
        
        self.assertIsInstance(stats, dict)
        for category, data in stats.items():
            self.assertIn('count', data)
            self.assertIn('total_amount', data)
            self.assertIn('avg_confidence', data)


class TestExpenseAnalyzer(unittest.TestCase):
    """Test cases for ExpenseAnalyzer."""
    
    def setUp(self):
        """Set up test data."""
        self.analyzer = ExpenseAnalyzer()
        self.test_transactions = [
            Transaction("001", "WALMART SUPERCENTER", 85.50, datetime.now() - timedelta(days=1), TransactionType.DEBIT),
            Transaction("002", "STARBUCKS COFFEE", 4.75, datetime.now() - timedelta(days=2), TransactionType.DEBIT),
            Transaction("003", "SALARY DEPOSIT", 3500.00, datetime.now() - timedelta(days=3), TransactionType.CREDIT),
            Transaction("004", "RENT PAYMENT", 1200.00, datetime.now() - timedelta(days=4), TransactionType.DEBIT),
        ]
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        self.assertIsInstance(self.analyzer, ExpenseAnalyzer)
        self.assertIsInstance(self.analyzer.classifier, CategoryClassifier)
        self.assertEqual(len(self.analyzer.transactions), 0)
        self.assertEqual(len(self.analyzer.categorized_transactions), 0)
    
    def test_add_single_transaction(self):
        """Test adding single transaction."""
        transaction = self.test_transactions[0]
        result = self.analyzer.add_transaction(transaction)
        
        self.assertEqual(len(self.analyzer.transactions), 1)
        self.assertEqual(len(self.analyzer.categorized_transactions), 1)
        self.assertIsInstance(result, CategorizationResult)
        self.assertEqual(result.transaction, transaction)
    
    def test_add_multiple_transactions(self):
        """Test adding multiple transactions."""
        results = self.analyzer.add_transactions(self.test_transactions)
        
        self.assertEqual(len(self.analyzer.transactions), len(self.test_transactions))
        self.assertEqual(len(self.analyzer.categorized_transactions), len(self.test_transactions))
        self.assertEqual(len(results), len(self.test_transactions))
    
    def test_analyze_expenses(self):
        """Test expense analysis."""
        self.analyzer.add_transactions(self.test_transactions)
        analysis = self.analyzer.analyze_expenses()
        
        self.assertIsInstance(analysis, ExpenseAnalysis)
        self.assertGreater(analysis.total_expenses, 0)
        self.assertGreater(analysis.total_income, 0)
        self.assertIsInstance(analysis.category_breakdown, dict)
        self.assertIsInstance(analysis.category_counts, dict)
        self.assertIsInstance(analysis.top_categories, list)
    
    def test_get_spending_trends(self):
        """Test spending trends."""
        self.analyzer.add_transactions(self.test_transactions)
        trends = self.analyzer.get_spending_trends(30)
        
        self.assertIsInstance(trends, dict)
    
    def test_get_category_insights(self):
        """Test category insights."""
        self.analyzer.add_transactions(self.test_transactions)
        insights = self.analyzer.get_category_insights(ExpenseCategory.GROCERIES)
        
        self.assertIsInstance(insights, dict)
        self.assertIn('category', insights)
        self.assertIn('total_amount', insights)
        self.assertIn('transaction_count', insights)
        self.assertIn('average_amount', insights)
        self.assertIn('confidence_avg', insights)
    
    def test_export_analysis(self):
        """Test analysis export."""
        self.analyzer.add_transactions(self.test_transactions)
        
        # Test JSON export
        json_export = self.analyzer.export_analysis("json")
        self.assertIsInstance(json_export, str)
        self.assertGreater(len(json_export), 0)
        
        # Test CSV export
        csv_export = self.analyzer.export_analysis("csv")
        self.assertIsInstance(csv_export, str)
        self.assertGreater(len(csv_export), 0)
    
    def test_clear_data(self):
        """Test data clearing."""
        self.analyzer.add_transactions(self.test_transactions)
        self.assertGreater(len(self.analyzer.transactions), 0)
        
        self.analyzer.clear_data()
        self.assertEqual(len(self.analyzer.transactions), 0)
        self.assertEqual(len(self.analyzer.categorized_transactions), 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Set up test data."""
        self.analyzer = ExpenseAnalyzer()
        self.sample_transactions = [
            Transaction("001", "WALMART SUPERCENTER", 85.50, datetime.now(), TransactionType.DEBIT),
            Transaction("002", "STARBUCKS COFFEE", 4.75, datetime.now(), TransactionType.DEBIT),
            Transaction("003", "SHELL GAS STATION", 52.30, datetime.now(), TransactionType.DEBIT),
            Transaction("004", "NETFLIX SUBSCRIPTION", 15.99, datetime.now(), TransactionType.DEBIT),
            Transaction("005", "SALARY DEPOSIT", 3500.00, datetime.now(), TransactionType.CREDIT),
            Transaction("006", "RENT PAYMENT", 1200.00, datetime.now(), TransactionType.DEBIT),
        ]
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Add transactions
        results = self.analyzer.add_transactions(self.sample_transactions)
        self.assertEqual(len(results), len(self.sample_transactions))
        
        # Verify categorization
        for result in results:
            self.assertIsNotNone(result.predicted_category)
            self.assertGreaterEqual(result.confidence_score, 0.0)
            self.assertLessEqual(result.confidence_score, 1.0)
        
        # Analyze expenses
        analysis = self.analyzer.analyze_expenses()
        self.assertGreater(analysis.total_expenses, 0)
        self.assertGreater(analysis.total_income, 0)
        
        # Get category insights
        for category in [ExpenseCategory.GROCERIES, ExpenseCategory.DINING, ExpenseCategory.TRANSPORTATION]:
            insights = self.analyzer.get_category_insights(category)
            self.assertIsInstance(insights, dict)
        
        # Export data
        json_export = self.analyzer.export_analysis("json")
        self.assertIsInstance(json_export, str)
        
        csv_export = self.analyzer.export_analysis("csv")
        self.assertIsInstance(csv_export, str)
    
    def test_performance_with_large_dataset(self):
        """Test performance with larger dataset."""
        # Create larger dataset
        large_transactions = []
        for i in range(100):
            transaction = Transaction(
                f"txn_{i:03d}",
                f"Transaction {i}",
                10.0 + i,
                datetime.now() - timedelta(days=i),
                TransactionType.DEBIT
            )
            large_transactions.append(transaction)
        
        # Test performance
        import time
        start_time = time.time()
        
        results = self.analyzer.add_transactions(large_transactions)
        analysis = self.analyzer.analyze_expenses()
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 100 transactions in reasonable time (< 5 seconds)
        self.assertLess(processing_time, 5.0)
        self.assertEqual(len(results), 100)
        self.assertIsInstance(analysis, ExpenseAnalysis)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestTransactionModel,
        TestCategoryRule,
        TestCategoryClassifier,
        TestExpenseAnalyzer,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("🧪 Running Personal Finance Expense Categorizer Tests")
    print("=" * 60)
    
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        exit(1)
