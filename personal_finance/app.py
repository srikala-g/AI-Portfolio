"""
Streamlit web application for the Personal Finance Expense Categorizer.
Provides an interactive interface for transaction categorization and analysis.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import List, Dict
import json

from models import Transaction, TransactionType, ExpenseCategory
from expense_analyzer import ExpenseAnalyzer
from category_classifier import CategoryClassifier


def initialize_session_state():
    """Initialize Streamlit session state."""
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = ExpenseAnalyzer()
    if 'transactions' not in st.session_state:
        st.session_state.transactions = []


def create_sample_data() -> List[Transaction]:
    """Create sample transactions for demo."""
    base_date = datetime.now() - timedelta(days=30)
    
    sample_transactions = [
        Transaction("txn_001", "WALMART SUPERCENTER", 85.50, base_date + timedelta(days=1), TransactionType.DEBIT),
        Transaction("txn_002", "STARBUCKS COFFEE", 4.75, base_date + timedelta(days=2), TransactionType.DEBIT),
        Transaction("txn_003", "SHELL GAS STATION", 52.30, base_date + timedelta(days=4), TransactionType.DEBIT),
        Transaction("txn_004", "NETFLIX SUBSCRIPTION", 15.99, base_date + timedelta(days=5), TransactionType.DEBIT),
        Transaction("txn_005", "AMAZON.COM PURCHASE", 67.89, base_date + timedelta(days=6), TransactionType.DEBIT),
        Transaction("txn_006", "ELECTRIC COMPANY BILL", 125.60, base_date + timedelta(days=8), TransactionType.DEBIT),
        Transaction("txn_007", "UBER TRIP", 18.75, base_date + timedelta(days=9), TransactionType.DEBIT),
        Transaction("txn_008", "RESTAURANT DINNER", 45.80, base_date + timedelta(days=10), TransactionType.DEBIT),
        Transaction("txn_009", "CVS PHARMACY", 23.45, base_date + timedelta(days=12), TransactionType.DEBIT),
        Transaction("txn_010", "SALARY DEPOSIT", 3500.00, base_date + timedelta(days=15), TransactionType.CREDIT),
        Transaction("txn_011", "RENT PAYMENT", 1200.00, base_date + timedelta(days=20), TransactionType.DEBIT),
        Transaction("txn_012", "HOTEL BOOKING", 189.99, base_date + timedelta(days=22), TransactionType.DEBIT),
    ]
    
    return sample_transactions


def display_transaction_form():
    """Display form for adding new transactions."""
    st.subheader("➕ Add New Transaction")
    
    with st.form("transaction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            description = st.text_input("Description", placeholder="e.g., WALMART SUPERCENTER")
            amount = st.number_input("Amount", min_value=0.01, step=0.01, format="%.2f")
            transaction_type = st.selectbox("Type", ["Debit", "Credit"])
        
        with col2:
            date = st.date_input("Date", value=datetime.now().date())
            merchant = st.text_input("Merchant (Optional)", placeholder="e.g., Walmart")
        
        submitted = st.form_submit_button("Add Transaction")
        
        if submitted:
            if description and amount > 0:
                transaction = Transaction(
                    id=f"txn_{len(st.session_state.transactions) + 1:03d}",
                    description=description,
                    amount=amount,
                    date=datetime.combine(date, datetime.min.time()),
                    transaction_type=TransactionType.DEBIT if transaction_type == "Debit" else TransactionType.CREDIT,
                    merchant=merchant if merchant else None
                )
                
                result = st.session_state.analyzer.add_transaction(transaction)
                st.session_state.transactions.append(transaction)
                
                st.success(f"Transaction added! Categorized as: {result.predicted_category.value.title()} "
                          f"(Confidence: {result.confidence_score:.2f})")
                st.rerun()
            else:
                st.error("Please fill in all required fields.")


def display_transactions_table():
    """Display transactions in a table."""
    st.subheader("📋 Transaction History")
    
    if not st.session_state.transactions:
        st.info("No transactions added yet. Add some transactions using the form above.")
        return
    
    # Create DataFrame for display
    data = []
    for i, transaction in enumerate(st.session_state.transactions):
        if i < len(st.session_state.analyzer.categorized_transactions):
            result = st.session_state.analyzer.categorized_transactions[i]
            data.append({
                'Date': transaction.date.strftime('%Y-%m-%d'),
                'Description': transaction.description,
                'Amount': f"${transaction.amount:.2f}",
                'Type': transaction.transaction_type.value.title(),
                'Category': result.predicted_category.value.title(),
                'Confidence': f"{result.confidence_score:.2f}",
                'Merchant': transaction.merchant or ''
            })
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)


def display_analysis_dashboard():
    """Display expense analysis dashboard."""
    st.subheader("📊 Expense Analysis Dashboard")
    
    if not st.session_state.transactions:
        st.info("Add some transactions to see the analysis dashboard.")
        return
    
    # Get analysis
    analysis = st.session_state.analyzer.analyze_expenses()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Expenses", f"${analysis.total_expenses:.2f}")
    
    with col2:
        st.metric("Total Income", f"${analysis.total_income:.2f}")
    
    with col3:
        st.metric("Net Balance", f"${analysis.net_balance:.2f}")
    
    with col4:
        st.metric("Uncategorized", analysis.uncategorized_count)
    
    # Category breakdown chart
    if analysis.category_breakdown:
        st.subheader("💰 Spending by Category")
        
        categories = list(analysis.category_breakdown.keys())
        amounts = list(analysis.category_breakdown.values())
        
        # Create pie chart
        fig = px.pie(
            values=amounts,
            names=[cat.value.title() for cat in categories],
            title="Expense Distribution by Category"
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        # Category breakdown table
        st.subheader("📋 Category Breakdown")
        breakdown_data = []
        for category, amount in analysis.top_categories:
            percentage = (amount / analysis.total_expenses) * 100 if analysis.total_expenses > 0 else 0
            breakdown_data.append({
                'Category': category.value.title(),
                'Amount': f"${amount:.2f}",
                'Percentage': f"{percentage:.1f}%",
                'Transactions': analysis.category_counts.get(category, 0)
            })
        
        breakdown_df = pd.DataFrame(breakdown_data)
        st.dataframe(breakdown_df, use_container_width=True)


def display_category_insights():
    """Display detailed category insights."""
    st.subheader("🔍 Category Insights")
    
    if not st.session_state.transactions:
        st.info("Add some transactions to see category insights.")
        return
    
    # Category selector
    available_categories = set()
    for result in st.session_state.analyzer.categorized_transactions:
        available_categories.add(result.predicted_category)
    
    if not available_categories:
        st.info("No categorized transactions available.")
        return
    
    selected_category = st.selectbox(
        "Select Category",
        options=sorted(available_categories, key=lambda x: x.value),
        format_func=lambda x: x.value.title()
    )
    
    if selected_category:
        insights = st.session_state.analyzer.get_category_insights(selected_category)
        
        if insights['transaction_count'] > 0:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Amount", f"${insights['total_amount']:.2f}")
            
            with col2:
                st.metric("Transaction Count", insights['transaction_count'])
            
            with col3:
                st.metric("Average Amount", f"${insights['average_amount']:.2f}")
            
            with col4:
                st.metric("Avg Confidence", f"{insights['confidence_avg']:.2f}")
            
            # Recent transactions
            if insights['recent_transactions']:
                st.subheader("Recent Transactions")
                recent_data = []
                for tx in insights['recent_transactions']:
                    recent_data.append({
                        'Date': tx['date'][:10],
                        'Description': tx['description'],
                        'Amount': f"${tx['amount']:.2f}",
                        'Confidence': f"{tx['confidence']:.2f}"
                    })
                
                recent_df = pd.DataFrame(recent_data)
                st.dataframe(recent_df, use_container_width=True)


def display_export_options():
    """Display data export options."""
    st.subheader("💾 Export Data")
    
    if not st.session_state.transactions:
        st.info("No data to export. Add some transactions first.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export as JSON"):
            json_data = st.session_state.analyzer.export_analysis("json")
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"expense_analysis_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("Export as CSV"):
            csv_data = st.session_state.analyzer.export_analysis("csv")
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"expense_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )


def main():
    """Main application function."""
    st.set_page_config(
        page_title="Personal Finance Expense Categorizer",
        page_icon="💰",
        layout="wide"
    )
    
    initialize_session_state()
    
    # Header
    st.title("💰 Personal Finance Expense Categorizer")
    st.markdown("Automatically categorize and analyze your financial transactions using AI-powered text classification.")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Add Transactions", "View Transactions", "Analysis Dashboard", "Category Insights", "Export Data"]
    )
    
    # Load sample data button
    if st.sidebar.button("Load Sample Data"):
        sample_transactions = create_sample_data()
        st.session_state.analyzer.add_transactions(sample_transactions)
        st.session_state.transactions.extend(sample_transactions)
        st.success(f"Loaded {len(sample_transactions)} sample transactions!")
        st.rerun()
    
    # Clear data button
    if st.sidebar.button("Clear All Data"):
        st.session_state.analyzer.clear_data()
        st.session_state.transactions.clear()
        st.success("All data cleared!")
        st.rerun()
    
    # Main content based on selected page
    if page == "Add Transactions":
        display_transaction_form()
    
    elif page == "View Transactions":
        display_transactions_table()
    
    elif page == "Analysis Dashboard":
        display_analysis_dashboard()
    
    elif page == "Category Insights":
        display_category_insights()
    
    elif page == "Export Data":
        display_export_options()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Features:**")
    st.sidebar.markdown("• Automatic categorization")
    st.sidebar.markdown("• Expense analysis")
    st.sidebar.markdown("• Category insights")
    st.sidebar.markdown("• Data export")
    st.sidebar.markdown("• Interactive dashboard")


if __name__ == "__main__":
    main()
