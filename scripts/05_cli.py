"""
=============================================================================
STW5000CEM — Introduction to Artificial Intelligence
Script 05: Command-Line Interface (CLI)
=============================================================================
Purpose : Interactive CLI to demonstrate the trained KNN classifier.
          Users can predict the price tier of any vegetable on any date.
Run     : python scripts/05_cli.py
          python scripts/05_cli.py --vegetable "Cauli Local" --date 2024-06-01 --price 45
=============================================================================
"""

import os, sys, json, pickle, argparse
import numpy as np
import pandas as pd
from datetime import datetime

MODEL_DIR = "model"

def load_model():
    def load(name):
        with open(os.path.join(MODEL_DIR, name), "rb") as f:
            return pickle.load(f)
    model     = load("best_model.pkl")
    scaler    = load("best_scaler.pkl")
    comm_enc  = load("commodity_encoder.pkl")
    label_enc = load("label_encoder.pkl")
    with open(os.path.join(MODEL_DIR, "last_prices.json")) as f:
        last_prices = json.load(f)
    with open(os.path.join(MODEL_DIR, "meta.json")) as f:
        meta = json.load(f)
    return model, scaler, comm_enc, label_enc, last_prices, meta

def build_features(vegetable, date_str, last_known_price, comm_enc, last_prices, feature_cols):
    d   = pd.Timestamp(date_str)
    enc = comm_enc.transform([vegetable])[0]
    lp_data = last_prices.get(vegetable, {})
    prices  = lp_data.get("prices", [50.0])
    if last_known_price: prices = list(prices) + [last_known_price]
    lag7   = prices[-7]  if len(prices) >= 7  else prices[0]
    lag30  = prices[-30] if len(prices) >= 30 else prices[0]
    roll7  = np.mean(prices[-7:]) if len(prices) >= 7 else np.mean(prices)
    row = {
        "CommodityEncoded": enc,
        "Year":       d.year, "Month": d.month, "DayOfYear": d.dayofyear,
        "WeekOfYear": int(d.isocalendar().week), "Quarter": (d.month-1)//3+1,
        "MonthSin":   np.sin(2*np.pi*d.month/12),
        "MonthCos":   np.cos(2*np.pi*d.month/12),
        "DaySin":     np.sin(2*np.pi*d.dayofyear/365),
        "DayCos":     np.cos(2*np.pi*d.dayofyear/365),
        "Lag7": lag7, "Lag30": lag30, "Roll7Mean": roll7, "PriceRange": 0.0,
    }
    return pd.DataFrame([row])[feature_cols].values

TIER_EMOJI = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Very High": "🔴"}
TIER_DESC  = {
    "Low":       "Below average price — good time to buy in bulk.",
    "Medium":    "Average market price — normal trading conditions.",
    "High":      "Above average price — consider alternatives.",
    "Very High": "Peak price — likely seasonal or supply shortage.",
}

def predict(vegetable, date_str, last_price, model, scaler, comm_enc, label_enc, last_prices, meta):
    X   = build_features(vegetable, date_str, last_price, comm_enc, last_prices, meta["feature_cols"])
    X_sc = scaler.transform(X)
    pred_encoded = model.predict(X_sc)[0]
    tier = label_enc.inverse_transform([pred_encoded])[0]
    proba = None
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X_sc)[0]
        proba = {label_enc.inverse_transform([i])[0]: round(float(p[i]), 3)
                 for i in range(len(p))}
    return tier, proba

def print_result(vegetable, date_str, tier, proba):
    print(f"\n  ┌──────────────────────────────────────────────────")
    print(f"  │  🥦 Vegetable  : {vegetable}")
    print(f"  │  📅 Date       : {date_str}")
    print(f"  │  {TIER_EMOJI.get(tier,'•')} Price Tier  : {tier}")
    print(f"  │  💡 Insight    : {TIER_DESC.get(tier,'')}")
    if proba:
        print(f"  │  📊 Probabilities:")
        for t in ["Low","Medium","High","Very High"]:
            if t in proba:
                bar = "█" * int(proba[t] * 20)
                print(f"  │     {t:<12}: {proba[t]:.3f}  {bar}")
    print(f"  └──────────────────────────────────────────────────\n")

def interactive_mode(model, scaler, comm_enc, label_enc, last_prices, meta):
    commodities = meta["commodities"]
    print("\n" + "="*55)
    print("  🥦 Kalimati Price Tier Classifier — CLI")
    print(f"  Model: {meta['best_model']}  |  Domain: Classification")
    print("="*55)
    print("  Commands: 'list' = show vegetables | 'quit' = exit\n")
    while True:
        print("-"*45)
        veg = input("  Vegetable name : ").strip()
        if veg.lower() in ("quit","exit","q"): print("  Goodbye! 👋"); break
        if veg.lower() == "list":
            for i,c in enumerate(commodities,1): print(f"  {i:2d}. {c}")
            continue
        if veg not in commodities:
            matches = [c for c in commodities if veg.lower() in c.lower()]
            if matches: veg = matches[0]; print(f"  (Matched: {veg})")
            else: print(f"  ❌ Not found. Type 'list'."); continue
        date_in = input("  Date (YYYY-MM-DD) [Enter=today] : ").strip()
        if not date_in: date_in = datetime.today().strftime("%Y-%m-%d")
        try: pd.Timestamp(date_in)
        except: print("  ❌ Invalid date. Use YYYY-MM-DD."); continue
        price_in = input("  Last known price (NPR/Kg) [Enter=auto] : ").strip()
        last_p   = float(price_in) if price_in else None
        tier, proba = predict(veg, date_in, last_p, model, scaler, comm_enc, label_enc, last_prices, meta)
        print_result(veg, date_in, tier, proba)

def main():
    parser = argparse.ArgumentParser(description="Kalimati Price Tier Classifier CLI")
    parser.add_argument("--vegetable", type=str, default=None)
    parser.add_argument("--date",      type=str, default=None)
    parser.add_argument("--price",     type=float, default=None)
    args = parser.parse_args()
    model, scaler, comm_enc, label_enc, last_prices, meta = load_model()
    if args.vegetable and args.date:
        tier, proba = predict(args.vegetable, args.date, args.price,
                              model, scaler, comm_enc, label_enc, last_prices, meta)
        print_result(args.vegetable, args.date, tier, proba)
    else:
        interactive_mode(model, scaler, comm_enc, label_enc, last_prices, meta)

if __name__ == "__main__":
    main()
