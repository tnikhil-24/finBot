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
        "hs_pattern_detector": ("👤 Head & Shoulders Analyst", "Reversal pattern diagnostician"),
        "llm_reason": ("🤖 LLM Market Strategist", "AI-powered market analysis synthesizer")
    }


    class State(TypedDict):
        messages: Annotated[list, add_messages]
        symbol: str
        ohlcv: pd.DataFrame
        analysis: str
        indicator_summary: str
        hs_pattern_signal: str
        hs_pattern_details: dict
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

    def head_shoulders_pattern_node(state):
        persona, role = NODE_PERSONAS["hs_pattern_detector"]
        df = state.get("ohlcv")
        if df is None or df.empty:
            state["hs_pattern_signal"] = "❌ No pattern data"
            state["hs_pattern_details"] = {}
            log_trace(state, "hs_pattern_detector", f"{persona} - No pattern data")
            return state

        prices = df["Close"].values
        df["local_max"] = df["Close"].iloc[argrelextrema(prices, np.greater_equal, order=3)[0]]
        df["local_min"] = df["Close"].iloc[argrelextrema(prices, np.less_equal, order=3)[0]]

        maxes = df.dropna(subset=["local_max"]).tail(7)
        mins = df.dropna(subset=["local_min"]).tail(7)

        signal = "⚠️ No head & shoulders pattern"
        details = {}
        close = df["Close"].iloc[-1]

        try:
            if len(maxes) >= 3:
                l, h, r = maxes["local_max"].values[-3:]
                if h > l and h > r and abs(l - r)/h < 0.05:
                    neckline = df["Close"].iloc[maxes.index[-1]+1:].min()
                    if close < neckline:
                        signal = "📉 Head & Shoulders → Bearish"
                        details = {"left": l, "head": h, "right": r, "neckline": neckline, "close": close}
            if len(mins) >= 3:
                l, h, r = mins["local_min"].values[-3:]
                if h < l and h < r and abs(l - r)/h < 0.05:
                    neckline = df["Close"].iloc[mins.index[-1]+1:].max()
                    if close > neckline:
                        signal = "📈 Inverse Head & Shoulders → Bullish"
                        details = {"left": l, "head": h, "right": r, "neckline": neckline, "close": close}
        except:
            pass

        state["hs_pattern_signal"] = signal
        state["hs_pattern_details"] = details
        log_trace(state, "hs_pattern_detector", f"{persona} {signal} | {details}")
        return state


    def backtest_llm_predictions():
        # Threshold parameters (customize these)
        entry_threshold = 1.0  # ENTER needs >1% gain to be "correct"
        exit_threshold = -1.0  # EXIT needs <1% loss to be "correct"
        volatility_threshold = 0.5  # HOLD needs <0.5% fluctuation

        # Load data with timestamp parsing
        full_data = pd.read_csv("D:/Projects/finBot/fetching/AAPL_4h_24mo_extended.csv",
                                parse_dates=['time'])
        test_cases = []

        # Window configuration (4h intervals)
        # window_size = 180 * 6  # 6 months
        # step_size = 14 * 6  # 2 weeks
        # holding_period = step_size  # Matching future window

        window_size = 90 * 6  # 3 months (better for momentum)
        step_size = 7 * 6  # 1 week (matches typical swing trades)
        holding_period = step_size

        print(f"🧪 Running backtest on {len(full_data)} periods...")

        for i in range(0, len(full_data) - window_size - holding_period, step_size):
            # Prepare historical data window
            hist_data = full_data.iloc[i:i + window_size].copy()
            hist_data.rename(columns={'time': 'Datetime'}, inplace=True)

            # Prepare future data window for price targets
            future_data = full_data.iloc[i + window_size:i + window_size + holding_period].copy()
            future_data.rename(columns={'time': 'Datetime'}, inplace=True)

            state = {
                "symbol": "AAPL",
                "ohlcv": hist_data
            }

            try:
                # Generate trading signal
                state = graph.invoke(state)
                signal = state.get("llm_trade_signal", {})

                # Price calculations
                entry_price = hist_data['close'].iloc[-1]  # Last price in historical window
                exit_price = future_data['close'].iloc[-1]  # Final price in future window

                # Return calculation
                actual_return = (exit_price / entry_price - 1) * 100  # Percentage return

                # Validation logic
                action = signal.get("action", "HOLD")
                if action == "ENTER":
                    correct = actual_return > entry_threshold  # Changed from >0
                elif action == "EXIT":
                    correct = actual_return < exit_threshold  # Changed from <0
                else:  # HOLD
                    correct = abs(actual_return) < volatility_threshold  # Changed from <1

                # Record complete trade details
                test_cases.append({
                    "entry_date": hist_data['Datetime'].iloc[-1].strftime('%Y-%m-%d'),
                    "entry_price": round(entry_price, 2),
                    "exit_date": future_data['Datetime'].iloc[-1].strftime('%Y-%m-%d'),
                    "exit_price": round(exit_price, 2),
                    "predicted_action": action,
                    "actual_return": round(actual_return, 2),
                    "correct": correct,
                    # Optional: Track thresholds used
                    "entry_threshold": entry_threshold,
                    "exit_threshold": exit_threshold,
                    "volatility_threshold": volatility_threshold
                })

            except Exception as e:
                print(f"⚠️ Error in window {i}: {str(e)}")
                continue

        # Performance metrics
        accuracy = np.mean([case['correct'] for case in test_cases])
        avg_return = np.mean([case['actual_return'] for case in test_cases])

        print(f"\n🔍 Backtest Complete ({len(test_cases)} trades analyzed)")
        print(f"Strategy Accuracy: {accuracy:.1%}")
        print(f"Average Return per Trade: {avg_return:.2f}%")

        # Save comprehensive results
        results_df = pd.DataFrame(test_cases)
        results_df.to_csv("pattern_backtest_results_v2.csv", index=False)
        print("\n💾 Saved detailed results to pattern_backtest_results.csv")

        return results_df

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
    builder.add_node("head_shoulders_pattern_node", head_shoulders_pattern_node)
    builder.add_node("llm_reason", llm_reason_node)

    builder.set_entry_point("api")
    builder.add_edge("api", "analyze")
    builder.add_edge("analyze", "indicators")
    builder.add_edge("indicators", "head_shoulders_pattern_node")
    builder.add_edge("head_shoulders_pattern_node", "llm_reason")
    builder.set_finish_point("llm_reason")
    langchain.debug = True

    graph = builder.compile(checkpointer=None)

    # Main execution loop
    if __name__ == "__main__":
            backtest_llm_predictions()
