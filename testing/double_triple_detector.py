import os
import sqlite3
import re

import pandas as pd
import numpy as np
import yfinance as yf
from langchain_core.tracers import langchain
from scipy.signal import argrelextrema
from dotenv import load_dotenv
import uuid
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
import google.generativeai as genai
from datetime import datetime

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Global in-memory database
memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
cursor = memory_conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        prompt TEXT,
        response TEXT
    )
''')
memory_conn.commit()

NODE_PERSONAS = {
    "api": ("📊 Data Curator", "Financial data acquisition specialist"),
    "analyze": ("📈 Trend Analyst", "Price movement pattern expert"),
    "indicators": ("📉 Indicator Specialist", "Technical indicator maestro"),
    "double_pattern_detector": ("🔍 Double Pattern Expert", "Double top/bottom authority"),
    "triple_pattern_detector": ("🔍 Triple Pattern Expert", "Triple formation specialist"),
    "llm_reason": ("🤖 LLM Market Strategist", "AI-powered market analysis synthesizer")
}


class State(TypedDict):
    messages: Annotated[list, add_messages]
    symbol: str
    ohlcv: pd.DataFrame
    analysis: str
    indicator_summary: str
    double_pattern_signal: str
    double_pattern_details: dict
    triple_pattern_signal: str
    triple_pattern_details: dict
    llm_opinion: str
    llm_prompt: str
    llm_trade_signal: dict
    response: str
    trace: list
    api_error: str


llm = genai.GenerativeModel("gemini-1.5-pro")


def save_interaction(prompt, response):
    cursor.execute('''
        INSERT INTO interactions (timestamp, prompt, response)
        VALUES (?, ?, ?)
    ''', (datetime.now().isoformat(), prompt, response))
    memory_conn.commit()


def log_trace(state, step_name, notes=None):
    state["trace"] = state.get("trace", [])
    persona, role = NODE_PERSONAS.get(step_name, ("🧑‍💻 Analyst", "Generalist"))
    state["trace"].append({
        "step": step_name,
        "persona": persona,
        "role": role,
        "timestamp": datetime.now().isoformat(),
        "input_keys": list(state.keys()),
        "summary": notes or "(no summary)"
    })


def extract_stock_symbol(user_input: str) -> str:
    prompt = f"What stock symbol is mentioned in this message: \"{user_input}\"? Respond with just the symbol, or say 'followup' if none."
    response = llm.generate_content(prompt).text.strip().upper()
    return response if response != "FOLLOWUP" else "FOLLOWUP"


def api_node(state):
    symbol = state["symbol"]
    try:
        file_path = "D:/Projects/finBot/fetching/AAPL_4h_24mo_extended.csv"
        df = pd.read_csv(file_path)

        # Ensure consistent column naming
        df.rename(columns={
            'time': 'Datetime',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)
        df.reset_index(drop=True, inplace=True)
        state["ohlcv"] = df
        log_trace(state, "api", f"Loaded {len(df)} rows from CSV for {symbol}")
    except Exception as e:
        state["ohlcv"] = pd.DataFrame()
        state["api_error"] = str(e)
        log_trace(state, "api", f"CSV loading error: {e}")
    return state


def analyze_node(state):
    persona, role = NODE_PERSONAS["analyze"]
    df = state.get("ohlcv")
    if df is None or df.empty:
        state["analysis"] = "❌ No price data"
        log_trace(state, "analyze", f"{persona} - No price data")
        return state
    closes = df["Close"].tail(5).tolist()
    trend = "↑" if closes[-1] > closes[0] else "↓"
    state["analysis"] = f"{persona} 5-hour trend: {trend} ({closes[0]:.2f} → {closes[-1]:.2f})"
    log_trace(state, "analyze", state["analysis"])
    return state


def indicator_node(state):
    persona, role = NODE_PERSONAS["indicators"]
    df = state.get("ohlcv")
    if df is None or df.empty:
        state["indicator_summary"] = "❌ No data"
        log_trace(state, "indicators", f"{persona} - No data")
        return state

    # Calculate technical indicators
    df["EMA_9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()
    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["VWAP"] = (df["Volume"] * (df["High"] + df["Low"] + df["Close"]) / 3).cumsum() / df["Volume"].cumsum()

    latest = df.dropna().iloc[-1]
    price = latest["Close"]

    state["indicator_summary"] = f"""
{persona} Indicators:

Price: {price:.2f}
VWAP: {latest['VWAP']:.2f} → {'Above' if price > latest['VWAP'] else 'Below'}
EMA 9/21: {latest['EMA_9']:.2f} / {latest['EMA_21']:.2f} → {'Bullish' if latest['EMA_9'] > latest['EMA_21'] else 'Bearish'}
RSI: {latest['RSI']:.2f} → {'Overbought' if latest['RSI'] > 70 else 'Oversold' if latest['RSI'] < 30 else 'Neutral'}
MACD: {latest['MACD']:.2f}, Signal: {latest['MACD_Signal']:.2f} → {'Bullish' if latest['MACD'] > latest['MACD_Signal'] else 'Bearish'}"""
    log_trace(state, "indicators", "Indicators calculated")
    return state


def double_pattern_detector_node(state):
    persona, role = NODE_PERSONAS["double_pattern_detector"]
    df = state.get("ohlcv")
    if df is None or df.empty:
        state["double_pattern_signal"] = "❌ No pattern data"
        return state

    prices = df["Close"].values
    max_idx = argrelextrema(prices, np.greater_equal, order=24)[0]
    min_idx = argrelextrema(prices, np.less_equal, order=24)[0]

    df["local_max"] = np.nan
    df["local_min"] = np.nan
    df.loc[max_idx, "local_max"] = df.loc[max_idx, "Close"]
    df.loc[min_idx, "local_min"] = df.loc[min_idx, "Close"]

    all_tops = df.dropna(subset=["local_max"])
    all_bottoms = df.dropna(subset=["local_min"])
    signal = "⚠️ No clear double pattern"
    details = {}
    close = df["Close"].iloc[-1]

    try:
        # Double Top detection
        for i in range(1, len(all_tops)):
            peak1 = all_tops.iloc[-i - 1]
            peak2 = all_tops.iloc[-i]
            if (peak2.name - peak1.name) > 180:
                continue
            top1 = peak1["local_max"]
            top2 = peak2["local_max"]
            peak_diff = abs(top2 - top1) / ((top1 + top2) / 2)
            trough = df["Close"].iloc[peak1.name:peak2.name].min()
            if peak_diff < 0.05 and close < trough:
                signal = "📉 Double Top → Bearish"
                details = {
                    "top1": top1,
                    "top2": top2,
                    "neckline": trough,
                    "peak_dates": [
                        df.iloc[peak1.name]["Datetime"].strftime("%Y-%m-%d"),
                        df.iloc[peak2.name]["Datetime"].strftime("%Y-%m-%d")
                    ],
                    "confirmation_date": df.iloc[-1]["Datetime"].strftime("%Y-%m-%d")
                }
                break

        # Double Bottom detection
        for i in range(1, len(all_bottoms)):
            trough1 = all_bottoms.iloc[-i - 1]
            trough2 = all_bottoms.iloc[-i]
            if (trough2.name - trough1.name) > 60:
                continue
            bottom1 = trough1["local_min"]
            bottom2 = trough2["local_min"]
            trough_diff = abs(bottom2 - bottom1) / ((bottom1 + bottom2) / 2)
            neckline = df["Close"].iloc[trough1.name:trough2.name].max()
            if trough_diff < 0.05 and close > neckline:
                signal = "📈 Double Bottom → Bullish"
                details = {
                    "bottom1": bottom1,
                    "bottom2": bottom2,
                    "neckline": neckline,
                    "trough_dates": [
                        df.iloc[trough1.name]["Datetime"].strftime("%Y-%m-%d"),
                        df.iloc[trough2.name]["Datetime"].strftime("%Y-%m-%d")
                    ],
                    "confirmation_date": df.iloc[-1]["Datetime"].strftime("%Y-%m-%d")
                }
                break

    except Exception as e:
        print(f"Pattern detection error: {str(e)}")

    state["double_pattern_signal"] = signal
    state["double_pattern_details"] = details
    log_trace(state, "double_pattern_detector", f"{persona} {signal} | {details}")
    return state


def triple_pattern_detector_node(state):
    persona, role = NODE_PERSONAS["triple_pattern_detector"]
    df = state.get("ohlcv")
    if df is None or df.empty:
        state["triple_pattern_signal"] = "❌ No pattern data"
        log_trace(state, "triple_pattern_detector", f"{persona} - No pattern data")
        return state

    prices = df["Close"].values
    max_idx = argrelextrema(prices, np.greater_equal, order=24)[0]
    min_idx = argrelextrema(prices, np.less_equal, order=24)[0]

    df["local_max"] = np.nan
    df["local_min"] = np.nan
    df.loc[max_idx, "local_max"] = df.loc[max_idx, "Close"]
    df.loc[min_idx, "local_min"] = df.loc[min_idx, "Close"]

    tops = df.dropna(subset=["local_max"]).tail(5)
    bottoms = df.dropna(subset=["local_min"]).tail(5)

    signal = "⚠️ No triple pattern"
    details = {}
    close = df["Close"].iloc[-1]

    try:
        # Triple Top detection
        if len(tops) >= 3:
            top1, top2, top3 = tops["local_max"].values[-3:]
            resistance = np.mean([top1, top2, top3])
            peak_dev = max(abs(top1 - resistance), abs(top2 - resistance), abs(top3 - resistance)) / resistance
            support = df["Close"].iloc[tops.index[-1] + 1:].min()
            if peak_dev < 0.02 and close < support:
                signal = "📉 Triple Top → Bearish"
                details = {
                    "top1": top1,
                    "top2": top2,
                    "top3": top3,
                    "resistance": resistance,
                    "support_broken": support,
                    "close": close
                }

        # Triple Bottom detection
        if len(bottoms) >= 3:
            bot1, bot2, bot3 = bottoms["local_min"].values[-3:]
            support = np.mean([bot1, bot2, bot3])
            trough_dev = max(abs(bot1 - support), abs(bot2 - support), abs(bot3 - support)) / support
            resistance = df["Close"].iloc[bottoms.index[-1] + 1:].max()
            if trough_dev < 0.02 and close > resistance:
                signal = "📈 Triple Bottom → Bullish"
                details = {
                    "bot1": bot1,
                    "bot2": bot2,
                    "bot3": bot3,
                    "support": support,
                    "resistance_broken": resistance,
                    "close": close
                }

    except Exception as e:
        print(f"Triple pattern error: {str(e)}")

    state["triple_pattern_signal"] = signal
    state["triple_pattern_details"] = details
    log_trace(state, "triple_pattern_detector", f"{persona} {signal} | {details}")
    return state


def backtest_llm_predictions():
    full_data = pd.read_csv("D:/Projects/finBot/fetching/AAPL_4h_24mo_extended.csv", parse_dates=['time'])
    test_cases = []

    window_size = 180 * 6  # 6 months in 4h intervals
    step_size = 14 * 6  # 2 weeks in 4h intervals

    print(f"🧪 Running backtest on {len(full_data)} periods...")

    for i in range(0, len(full_data) - window_size - step_size, step_size):
        # 1. RENAME COLUMNS HERE FIRST
        hist_data = full_data.iloc[i:i + window_size].copy()
        hist_data.rename(columns={'time': 'Datetime'}, inplace=True)  # Fix 1: Rename early

        future_data = full_data.iloc[i + window_size:i + window_size + step_size].copy()
        future_data.rename(columns={'time': 'Datetime'}, inplace=True)  # Fix 2: For consistency

        state = {
            "symbol": "AAPL",
            "ohlcv": hist_data  # Now contains 'Datetime' column
        }

        # 2. FIX DATE ACCESS
        try:
            state = graph.invoke(state)
            signal = state.get("llm_trade_signal", {})

            # 3. USE RENAMED COLUMN
            actual_return = (future_data['close'].iloc[-1] / future_data['close'].iloc[0] - 1) * 100

            test_cases.append({
                "start_date": hist_data['Datetime'].iloc[0].strftime('%Y-%m-%d'),  # Now valid
                "end_date": hist_data['Datetime'].iloc[-1].strftime('%Y-%m-%d'),
                "predicted_action": signal.get("action", "HOLD"),
                "actual_return": actual_return,
                "correct": (
                        (signal.get("action") == "ENTER" and actual_return > 0) or
                        (signal.get("action") == "EXIT" and actual_return < 0) or
                        (signal.get("action") == "HOLD" and abs(actual_return) < 1)
                )
            })

        except Exception as e:
            print(f"Error in window {i}: {str(e)}")

    accuracy = sum(case['correct'] for case in test_cases) / len(test_cases)
    avg_return = np.mean([case['actual_return'] for case in test_cases])

    print(f"\n🔍 Backtest Complete ({len(test_cases)} test cases)")
    print(f"Final Accuracy: {accuracy:.1%}")
    print(f"Average Return: {avg_return:.2f}%")

    # Save results to CSV
    pd.DataFrame(test_cases).to_csv("pattern_backtest_results.csv", index=False)
    print("\n💾 Saved results to pattern_backtest_results.csv")
    return test_cases

def llm_reason_node(state):
    persona, role = NODE_PERSONAS["llm_reason"]
    df = state.get("ohlcv")
    if df is None or df.empty:
        state["llm_opinion"] = "❌ No OHLCV data to analyze"
        state["llm_prompt"] = ""
        log_trace(state, "llm_reason", f"{persona} - Skipped due to missing data")
        return state

    ohlcv_data = df.tail(50).to_dict(orient="records")

    pattern_debug = f"""
Double: {state.get("double_pattern_signal", "")}
Triple: {state.get("triple_pattern_signal", "")}
"""

    indicators_raw = df[["EMA_9", "EMA_21", "RSI", "MACD", "MACD_Signal", "VWAP"]].dropna().tail(50).to_dict(
        orient="list")

    prompt = f"""
    You are {persona} ({role}).

    📌 SYMBOL: {state['symbol']}

    Patterns Detected:
    - Double: {state.get('double_pattern_signal')}
    - Triple: {state.get('triple_pattern_signal')}

    Key Indicators:
    - Price: {df['Close'].iloc[-1]:.2f}
    - RSI: {df['RSI'].iloc[-1]:.1f}
    - MACD Histogram: {df['MACD'].iloc[-1] - df['MACD_Signal'].iloc[-1]:.3f}
    - EMA 9/21: {df['EMA_9'].iloc[-1]:.2f}/{df['EMA_21'].iloc[-1]:.2f}

    Should we ENTER long, EXIT positions, or HOLD?
    - ENTER if bullish pattern + indicators confirm
    - EXIT if bearish pattern + indicators warn
    - HOLD if conflicting signals

    Format response as:
    Action: [ENTER/EXIT/HOLD]
    Entry Price: [price or N/A]
    Reason: [2-sentence explanation]
    """.strip()

    result = llm.generate_content(prompt).text.strip()
    state["llm_opinion"] = result
    state["llm_prompt"] = prompt

    # Parse LLM response
    match = re.search(
        r"Action:\s*(\w+).*Entry Price:\s*([^\n]+).*Exit Price:\s*([^\n]+).*Reason:\s*(.*)",
        result,
        re.DOTALL
    )

    if match:
        state["llm_trade_signal"] = {
            "action": match.group(1).strip(),
            "entry_price": match.group(2).strip(),
            "exit_price": match.group(3).strip(),
            "reason": match.group(4).strip()
        }
    else:
        state["llm_trade_signal"] = {
            "action": "HOLD",
            "entry_price": "N/A",
            "exit_price": "N/A",
            "reason": "No clear signal detected"
        }

    log_trace(state, "llm_reason", f"{persona} LLM result: {result}")
    save_interaction(prompt, result)
    return state


# Build and configure the state graph
builder = StateGraph(State)
builder.add_node("api", api_node)
builder.add_node("analyze", analyze_node)
builder.add_node("indicators", indicator_node)
builder.add_node("double_pattern_detector", double_pattern_detector_node)
builder.add_node("triple_pattern_detector", triple_pattern_detector_node)
builder.add_node("llm_reason", llm_reason_node)

builder.set_entry_point("api")
builder.add_edge("api", "analyze")
builder.add_edge("analyze", "indicators")
builder.add_edge("indicators", "double_pattern_detector")
builder.add_edge("double_pattern_detector", "triple_pattern_detector")
builder.add_edge("triple_pattern_detector", "llm_reason")
builder.set_finish_point("llm_reason")
langchain.debug = True

graph = builder.compile(checkpointer=None)

# Main execution loop
if __name__ == "__main__":
        backtest_llm_predictions()
