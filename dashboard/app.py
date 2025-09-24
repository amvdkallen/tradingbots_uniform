import os
import sqlite3
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def get_db_path():
    return os.getenv('DB_PATH', 'tradingbots.db')

def load_data():
    db_path = get_db_path()
    
    if not os.path.exists(db_path):
        return None, None, None
    
    try:
        conn = sqlite3.connect(db_path)
        
        events_query = "SELECT * FROM events ORDER BY id DESC LIMIT 100"
        events_df = pd.read_sql_query(events_query, conn)
        
        equity_query = "SELECT * FROM equity ORDER BY id"
        equity_df = pd.read_sql_query(equity_query, conn)
        
        fills_query = "SELECT * FROM events WHERE event_type = 'fill' ORDER BY id DESC LIMIT 50"
        fills_df = pd.read_sql_query(fills_query, conn)
        
        conn.close()
        return events_df, equity_df, fills_df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

def main():
    st.set_page_config(page_title="Trading Bot Dashboard", layout="wide")
    st.title("Trading Bot Dashboard")
    
    events_df, equity_df, fills_df = load_data()
    
    if events_df is None or events_df.empty:
        st.info("No data available. Please run a bot first.")
        return
    
    bots = events_df['bot'].unique()
    selected_bot = st.sidebar.selectbox("Select Bot", bots)
    
    bot_events = events_df[events_df['bot'] == selected_bot]
    bot_equity = equity_df[equity_df['bot'] == selected_bot] if not equity_df.empty else pd.DataFrame()
    bot_fills = fills_df[fills_df['bot'] == selected_bot] if not fills_df.empty else pd.DataFrame()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Equity Curve")
        if not bot_equity.empty:
            bot_equity['ts'] = pd.to_datetime(bot_equity['ts'])
            st.line_chart(bot_equity.set_index('ts')['equity'])
        else:
            st.info("No equity data available")
    
    with col2:
        st.subheader("Position")
        if not bot_equity.empty:
            latest = bot_equity.iloc[-1]
            st.metric("Current Equity", f"${latest['equity']:.2f}")
            st.metric("Position Size", f"{latest['position_qty']:.4f}")
            if latest['position_avg_price'] > 0:
                st.metric("Avg Price", f"${latest['position_avg_price']:.4f}")
    
    st.subheader("Recent Events")
    if not bot_events.empty:
        display_cols = ['ts', 'event_type', 'side', 'qty', 'price', 'reason']
        available_cols = [col for col in display_cols if col in bot_events.columns]
        st.dataframe(bot_events[available_cols].head(20))
    
    st.subheader("Recent Fills")
    if not bot_fills.empty:
        display_cols = ['ts', 'side', 'qty', 'price', 'reason']
        available_cols = [col for col in display_cols if col in bot_fills.columns]
        st.dataframe(bot_fills[available_cols])
    else:
        st.info("No fills yet")

if __name__ == "__main__":
    main()
