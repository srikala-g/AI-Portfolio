"""
Demo application for the Personal Finance Expense Categorizer.
Shows how to use the system with sample transactions.
"""

import json
from datetime import datetime, timedelta
from typing import List

from models import Transaction, TransactionType, ExpenseCategory
from expense_analyzer import ExpenseAnalyzer
from category_classifier import CategoryClassifier


def create_sample_transactions() -> List[Transaction]:
    """Create sample transactions for demonstration."""
    base_date = datetime.now() - timedelta(days=30)
    
    sample_transactions = [
        # Groceries
        Transaction(
            id="txn_001",
            description="WALMART SUPERCENTER #1234",
            amount=85.50,
            date=base_date + timedelta(days=1),
            transaction_type=TransactionType.DEBIT,
            merchant="Walmart"
        ),
        Transaction(
            id="txn_002", 
            description="WHOLE FOODS MARKET",
            amount=127.30,
            date=base_date + timedelta(days=3),
            transaction_type=TransactionType.DEBIT,
            merchant="Whole Foods"
        ),
        
        # Dining
        Transaction(
            id="txn_003",
            description="STARBUCKS COFFEE #4567",
            amount=4.75,
            date=base_date + timedelta(days=2),
            transaction_type=TransactionType.DEBIT,
            merchant="Starbucks"
        ),
        Transaction(
            id="txn_004",
            description="MCDONALD'S #8901",
            amount=12.45,
            date=base_date + timedelta(days=5),
            transaction_type=TransactionType.DEBIT,
            merchant="McDonald's"
        ),
        Transaction(
            id="txn_005",
            description="ITALIAN RESTAURANT",
            amount=45.80,
            date=base_date + timedelta(days=7),
            transaction_type=TransactionType.DEBIT,
            merchant="Local Restaurant"
        ),
        
        # Transportation
        Transaction(
            id="txn_006",
            description="SHELL GAS STATION",
            amount=52.30,
            date=base_date + timedelta(days=4),
            transaction_type=TransactionType.DEBIT,
            merchant="Shell"
        ),
        Transaction(
            id="txn_007",
            description="UBER TRIP",
            amount=18.75,
            date=base_date + timedelta(days=6),
            transaction_type=TransactionType.DEBIT,
            merchant="Uber"
        ),
        
        # Utilities
        Transaction(
            id="txn_008",
            description="ELECTRIC COMPANY BILL",
            amount=125.60,
            date=base_date + timedelta(days=8),
            transaction_type=TransactionType.DEBIT,
            merchant="Electric Company"
        ),
        Transaction(
            id="txn_009",
            description="INTERNET SERVICE PROVIDER",
            amount=79.99,
            date=base_date + timedelta(days=10),
            transaction_type=TransactionType.DEBIT,
            merchant="ISP"
        ),
        
        # Entertainment
        Transaction(
            id="txn_010",
            description="NETFLIX SUBSCRIPTION",
            amount=15.99,
            date=base_date + timedelta(days=12),
            transaction_type=TransactionType.DEBIT,
            merchant="Netflix"
        ),
        Transaction(
            id="txn_011",
            description="MOVIE THEATER TICKETS",
            amount=24.50,
            date=base_date + timedelta(days=14),
            transaction_type=TransactionType.DEBIT,
            merchant="AMC Theater"
        ),
        
        # Shopping
        Transaction(
            id="txn_012",
            description="AMAZON.COM PURCHASE",
            amount=67.89,
            date=base_date + timedelta(days=9),
            transaction_type=TransactionType.DEBIT,
            merchant="Amazon"
        ),
        Transaction(
            id="txn_013",
            description="TARGET STORE #5678",
            amount=89.45,
            date=base_date + timedelta(days=11),
            transaction_type=TransactionType.DEBIT,
            merchant="Target"
        ),
        
        # Healthcare
        Transaction(
            id="txn_014",
            description="CVS PHARMACY #123",
            amount=23.45,
            date=base_date + timedelta(days=13),
            transaction_type=TransactionType.DEBIT,
            merchant="CVS"
        ),
        Transaction(
            id="txn_015",
            description="DOCTOR'S OFFICE VISIT",
            amount=150.00,
            date=base_date + timedelta(days=15),
            transaction_type=TransactionType.DEBIT,
            merchant="Medical Center"
        ),
        
        # Travel
        Transaction(
            id="txn_016",
            description="HOTEL BOOKING.COM",
            amount=189.99,
            date=base_date + timedelta(days=16),
            transaction_type=TransactionType.DEBIT,
            merchant="Booking.com"
        ),
        Transaction(
            id="txn_017",
            description="AIRLINE TICKET",
            amount=450.00,
            date=base_date + timedelta(days=18),
            transaction_type=TransactionType.DEBIT,
            merchant="Delta Airlines"
        ),
        
        # Income
        Transaction(
            id="txn_018",
            description="SALARY DEPOSIT",
            amount=3500.00,
            date=base_date + timedelta(days=20),
            transaction_type=TransactionType.CREDIT,
            merchant="Employer"
        ),
        Transaction(
            id="txn_019",
            description="FREELANCE PAYMENT",
            amount=500.00,
            date=base_date + timedelta(days=22),
            transaction_type=TransactionType.CREDIT,
            merchant="Client"
        ),
        
        # Rent
        Transaction(
            id="txn_020",
            description="RENT PAYMENT",
            amount=1200.00,
            date=base_date + timedelta(days=25),
            transaction_type=TransactionType.DEBIT,
            merchant="Landlord"
        ),
        
        # Other/Unclear
        Transaction(
            id="txn_021",
            description="ATM WITHDRAWAL",
            amount=100.00,
            date=base_date + timedelta(days=26),
            transaction_type=TransactionType.DEBIT,
            merchant="Bank ATM"
        ),
        Transaction(
            id="txn_022",
            description="UNKNOWN CHARGE",
            amount=25.00,
            date=base_date + timedelta(days=28),
            transaction_type=TransactionType.DEBIT,
            merchant="Unknown"
        )
    ]
    
    return sample_transactions


def run_demo():
    """Run the expense categorizer demo."""
    print("🏦 Personal Finance Expense Categorizer Demo")
    print("=" * 50)
    
    # Create analyzer
    analyzer = ExpenseAnalyzer()
    
    # Load sample transactions
    print("\n📊 Loading sample transactions...")
    sample_transactions = create_sample_transactions()
    print(f"Loaded {len(sample_transactions)} sample transactions")
    
    # Add transactions to analyzer
    print("\n🔍 Categorizing transactions...")
    results = analyzer.add_transactions(sample_transactions)
    
    # Show categorization results
    print("\n📋 Categorization Results:")
    print("-" * 80)
    print(f"{'Date':<12} {'Description':<25} {'Amount':<10} {'Category':<15} {'Confidence':<10}")
    print("-" * 80)
    
    for result in sorted(results, key=lambda x: x.transaction.date):
        transaction = result.transaction
        print(f"{transaction.date.strftime('%Y-%m-%d'):<12} "
              f"{transaction.description[:24]:<25} "
              f"${transaction.amount:<9.2f} "
              f"{result.predicted_category.value:<15} "
              f"{result.confidence_score:<9.2f}")
    
    # Run analysis
    print("\n📈 Expense Analysis:")
    print("-" * 40)
    analysis = analyzer.analyze_expenses()
    
    print(f"Total Expenses: ${analysis.total_expenses:.2f}")
    print(f"Total Income: ${analysis.total_income:.2f}")
    print(f"Net Balance: ${analysis.net_balance:.2f}")
    print(f"Uncategorized Transactions: {analysis.uncategorized_count}")
    
    # Category breakdown
    print("\n💰 Category Breakdown:")
    print("-" * 40)
    for category, amount in analysis.top_categories:
        percentage = (amount / analysis.total_expenses) * 100 if analysis.total_expenses > 0 else 0
        print(f"{category.value.title():<15}: ${amount:<8.2f} ({percentage:.1f}%)")
    
    # Category insights
    print("\n🔍 Detailed Category Insights:")
    print("-" * 50)
    
    for category in [ExpenseCategory.GROCERIES, ExpenseCategory.DINING, 
                   ExpenseCategory.TRANSPORTATION, ExpenseCategory.ENTERTAINMENT]:
        insights = analyzer.get_category_insights(category)
        if insights['transaction_count'] > 0:
            print(f"\n{category.value.title()}:")
            print(f"  Total Amount: ${insights['total_amount']:.2f}")
            print(f"  Transactions: {insights['transaction_count']}")
            print(f"  Average: ${insights['average_amount']:.2f}")
            print(f"  Confidence: {insights['confidence_avg']:.2f}")
    
    # Export sample
    print("\n💾 Exporting data...")
    json_export = analyzer.export_analysis("json")
    print(f"JSON export size: {len(json_export)} characters")
    
    # Show spending trends
    print("\n📊 Spending Trends (Last 30 days):")
    print("-" * 40)
    trends = analyzer.get_spending_trends(30)
    
    for category, daily_amounts in trends.items():
        if daily_amounts:
            total_spent = sum(amount for _, amount in daily_amounts)
            print(f"{category.title()}: ${total_spent:.2f} over {len(daily_amounts)} days")
    
    print("\n✅ Demo completed successfully!")
    print("\n💡 Key Features Demonstrated:")
    print("  • Automatic transaction categorization")
    print("  • Expense analysis and reporting")
    print("  • Category-based insights")
    print("  • Data export capabilities")
    print("  • Spending trend analysis")


if __name__ == "__main__":
    run_demo()
