import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from langchain.llms import CTransformers


# Define a function to get financial data
def get_financial_data(ticker):
    stock = yf.Ticker(ticker)
    try:
        balance_sheet = stock.balance_sheet
        income_statement = stock.financials
        cash_flow = stock.cashflow
        return balance_sheet, income_statement, cash_flow
    except Exception as e:
        return None, None, None

# Define a function to get response from LLama 2 model
def getLLamaresponse(prompt_text):
    # Initialize LLama 2 model
    llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                        model_type='llama',
                        config={'max_new_tokens': 256,
                                'temperature': 0.01})
    
    # Generate the response from the LLama 2 model
    response = llm(prompt_text)
    
    # Split the response into chunks of 1000 characters
    response_chunks = [response[i:i+1000] for i in range(0, len(response), 1000)]
    
    return response_chunks

# Streamlit app configuration
st.set_page_config(page_title="Financial Insights Agent",
                   page_icon='📈',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("Financial Insights Agent 📈")

# Input fields for ticker symbol and company value proposition
ticker = st.text_input("Enter the Ticker Symbol (e.g., AAPL, MSFT)")
value_proposition = st.text_area("Enter the Company's Unique Value Proposition")

# Submit button to generate insights
submit = st.button("Generate Financial Insights")

if submit and ticker and value_proposition:
    # Fetch and display financial data
    st.subheader("Financial Data")
    balance_sheet, income_statement, cash_flow = get_financial_data(ticker)
    
    if balance_sheet is not None and income_statement is not None and cash_flow is not None:
        st.write("Balance Sheet")
        st.write(balance_sheet)
        st.write("Income Statement")
        st.write(income_statement)
        st.write("Cash Flow")
        st.write(cash_flow)
        
        # Create prompts for LLama model
        sections = ["Earnings Data Analysis", "Financial Data Analysis", "Brainstorm Values"]
        for section in sections:
            prompt = f"""
            Section: {section}
            Company: {ticker}
            Value Proposition: {value_proposition}
            
            Analyze the information and provide detailed insights.
            """
            st.subheader(section)
            response_chunks = getLLamaresponse(prompt)
            
            # Display response chunks iteratively
            for chunk in response_chunks:
                st.write(chunk)
        
        # Plot financial data
        st.subheader("Financial Data Visualization")
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        balance_sheet.plot(ax=axes[0], kind='line', title='Balance Sheet')
        income_statement.plot(ax=axes[1], kind='line', title='Income Statement')
        cash_flow.plot(ax=axes[2], kind='line', title='Cash Flow')
        
        # Display plots
        st.pyplot(fig)
    else:
        st.write("Error fetching financial data.")