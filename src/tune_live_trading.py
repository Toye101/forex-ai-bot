import itertools
from simulate_live_trading import simulate_live_trading

def tune_parameters():
    atr_multipliers = [0.3, 0.5, 0.7, 1.0]
    risks = [0.05, 0.01, 0.15]
    reward_to_risks = [2.0, 3.0, 4.0]
    min_probs = [0.55, 0.6, 0.65]

    best = None
    best_balance = 0

    print("\n🚀 Starting parameter tuning sweep...\n")

    for atr_mult, risk, rr, prob in itertools.product(atr_multipliers, risks, reward_to_risks, min_probs):
        print(f"Testing: ATRx{atr_mult}, Risk={risk}, RR={rr}, Prob>={prob}")
        try:
            results = simulate_live_trading(
                atr_mult=atr_mult,
                risk_per_trade=risk,
                reward_to_risk=rr,
                min_prob_threshold=prob,
            )

            final_balance = results['final_balance']
            print(f"✅ Result: Balance={final_balance:.2f}, WinRate={results['win_rate']:.2%}, ProfitFactor={results['profit_factor']:.2f}")

            if final_balance > best_balance:
                best_balance = final_balance
                best = (atr_mult, risk, rr, prob, results)

        except Exception as e:
            print(f"⚠ Error during test: {e}")

    if best:
        atr_mult, risk, rr, prob, results = best
        print("\n🏆 Best Parameters Found:")
        print(f"ATR Multiplier: {atr_mult}")
        print(f"Risk per Trade: {risk}")
        print(f"Reward-to-Risk: {rr}")
        print(f"Min Prob Threshold: {prob}")
        print(f"Final Balance: {results['final_balance']:.2f}")
        print(f"Win Rate: {results['win_rate']:.2%}")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
    else:
        print("❌ No valid results found.")

if __name__ == "__main__":
    tune_parameters()