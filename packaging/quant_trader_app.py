"""Entry point of QuantTrader.exe (see QuantTrader.spec)."""

import sys

from quant_trader.app import main

if __name__ == "__main__":
    sys.exit(main())
