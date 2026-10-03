market-scanner-options is a single-file Python tool. It reads an option-chain CSV (Opstra-style columns work as they are) and finds strategies that give an "all-green" payoff graph: a profit above ₹0 at every underlying price at expiry. It checks box spreads, vertical credit and debit spreads, and 4-leg two-call-plus-two-put combinations.

Works for any stock or index. Only the CSV changes. Optional filters for low open interest and stale prices (put-call parity check). Can price legs at LTP, or at Ask for buys and Bid for sells (--prices bidask). A solve mode takes your own 4 legs and shows the price one leg would need for a green payoff.

Command Line Usage & Help
Usage:
  market_scanner.exe <CSV_FILE>
  python market_scanner.py <CSV_FILE>

Arguments:
  <CSV_FILE>                Path to the input option chain CSV file (e.g., chain_data.csv or "C:\path\to\chain.csv").

Options:
  -h, --help                Show this help message and exit.
  

Disclaimer: this is an educational tool and is not financial advice. Apparent all-green setups at last-traded prices are usually caused by stale quotes, bid-ask spreads or illiquid strikes. Check live bid/ask prices, charges and settlement rules (stock options in India are physically settled) before trading.
