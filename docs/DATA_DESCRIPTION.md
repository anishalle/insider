# Polymarket Dataset Description

Complete guide to the 5 parquet files in this dataset.

---

## 📁 File Overview

| File | Size | Description | Use Case |
|------|------|-------------|----------|
| `orderfilled.parquet` | 31GB | Raw blockchain events | Complete historical record |
| `trades.parquet` | 32GB | Processed trading data | Market analysis, price tracking |
| `markets.parquet` | 68MB | Market metadata | Market info, token mapping |
| `quant.parquet` | 21GB | Quantitative trading data | Trading algorithms, backtesting |
| `users.parquet` | 23GB | User behavior data | User analysis, wallet tracking |

---

## 1️⃣ orderfilled.parquet (31GB)

**Raw OrderFilled events from Polygon blockchain**

### Description
Original blockchain events decoded from two Polymarket exchange contracts:
- CTF Exchange: `0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e`
- NegRisk CTF Exchange: `0xc5d563a36ae78145c45a50134d48a1215220f80a`

### Schema
```
timestamp           int64      # Unix timestamp
datetime            string     # Human-readable datetime
block_number        int64      # Blockchain block number
transaction_hash    string     # Transaction hash
log_index           int64      # Log index within transaction
contract            string     # Contract name
maker               string     # Maker address
taker               string     # Taker address
maker_asset_id      string     # Maker's asset ID (uint256 as string)
taker_asset_id      string     # Taker's asset ID (uint256 as string)
maker_amount_filled int64      # Maker's filled amount (wei)
taker_amount_filled int64      # Taker's filled amount (wei)
maker_fee           int64      # Maker fee (wei)
taker_fee           int64      # Taker fee (wei)
protocol_fee        int64      # Protocol fee (wei)
order_hash          string     # Order hash
```

### Key Features
- ✅ Complete blockchain data with no missing fields
- ✅ Includes block_number for time-series analysis
- ✅ Includes all fees (maker, taker, protocol)
- ✅ Contains order_hash for order tracking

### Use Cases
- Full historical blockchain analysis
- Fee analysis
- Order matching studies
- Raw event processing

---

## 2️⃣ trades.parquet (32GB)

**Processed trading data with market linkage**

### Description
Enhanced version of orderfilled events with:
- Market ID linkage (from token to market)
- Trade direction analysis (BUY/SELL)
- Price calculation
- USD amount extraction

### Schema
All fields from `orderfilled.parquet`, plus:
```
market_id           string     # Linked market ID
condition_id        string     # Condition ID
token_id            string     # Non-USDC token ID
answer              string     # Market outcome (YES/NO/etc.)
nonusdc_side        string     # Which side has token (token1/token2)
maker_direction     string     # Maker's direction (BUY/SELL)
taker_direction     string     # Taker's direction (BUY/SELL)
price               float64    # Trade price (0-1)
token_amount        int64      # Token amount (in wei)
usd_amount          int64      # USDC amount (in wei)
```

### Key Features
- ✅ Auto-linked to market metadata
- ✅ Trade direction calculated for both sides
- ✅ Price normalized to 0-1 range
- ✅ Preserves all original blockchain fields

### Use Cases
- Market price analysis
- Trading volume by market
- Direction-based analysis
- General trading analytics

---

## 3️⃣ markets.parquet (68MB)

**Market metadata from Gamma API**

### Description
Complete market information including:
- Market questions
- Outcome names
- Token IDs
- Resolution status

### Schema
```
market_id           string     # Unique market ID
question            string     # Market question
description         string     # Market description
outcomes            list       # List of outcome names
tokens              list       # List of token IDs
volume              float64    # Total trading volume
liquidity           float64    # Market liquidity
resolved            bool       # Is market resolved?
resolution          string     # Resolution outcome
created_at          timestamp  # Market creation time
end_date            timestamp  # Market end date
```

### Use Cases
- Market information lookup
- Token ID to market mapping
- Market category analysis

---

## 4️⃣ quant.parquet (21GB) - Quantitative Trading Data

**Cleaned trades with unified token perspective and direction**

### Description
Optimized version of `trades.parquet` for quantitative analysis with:
1. **Unified Token Perspective**: All trades normalized to YES token (token1)
2. **Unified Direction**: All trades converted to BUY perspective
3. **Contract Filtering**: Removed contract-to-contract trades

### Data Cleaning Process

#### Step 1: Filter Invalid Data
```python
# Remove trades with NaN prices
df = df[~df['price'].isna()]

# Remove contract addresses as taker (internal contract operations)
contract_addresses = {
    '0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e',  # CTF Exchange
    '0xc5d563a36ae78145c45a50134d48a1215220f80a'   # NegRisk CTF Exchange
}
df = df[~df['taker'].str.lower().isin(contract_addresses)]
```

#### Step 2: Unified Token Perspective (YES/token1)
```python
# If trade involves token2 (NO token), flip the price
is_token2 = df['nonusdc_side'] == 'token2'
df.loc[is_token2, 'price'] = 1 - df.loc[is_token2, 'price']
df['nonusdc_side'] = 'token1'  # All normalized to token1/YES

# Example:
# Original: token2, price=0.3 → Normalized: token1, price=0.7
```

**Why?** In prediction markets:
- token1 = YES outcome (probability)
- token2 = NO outcome (1 - probability)

By normalizing all trades to YES perspective:
- price = 0.7 means 70% probability of YES
- Easier to compare across all trades
- Consistent time-series analysis

#### Step 3: Keep All Original Fields
Unlike `users.parquet`, `quant.parquet` preserves:
- Both maker and taker addresses
- Both maker_direction and taker_direction
- All blockchain metadata

### Schema
Same as `trades.parquet`, but with:
```
nonusdc_side        string     # Always 'token1' (unified)
price               float64    # YES token price (unified)
```

### Key Features
- ✅ Unified YES token perspective for all trades
- ✅ Contract trades filtered out
- ✅ Maintains maker/taker structure
- ✅ Preserves all original fields
- ✅ Clean data for backtesting

### Use Cases
- **Algorithmic Trading**: Consistent price signals
- **Backtesting**: Unified direction simplifies strategy testing
- **Price Prediction**: Model YES probability trends
- **Market Making**: Analyze bid-ask spreads
- **Volume Analysis**: Track real trading activity (no contracts)

---

## 5️⃣ users.parquet (23GB) - User Behavior Data

**User-centric view with split maker/taker records**

### Description
Transformed view of trades optimized for user behavior analysis:
1. **Split Records**: Each trade becomes 2 records (maker + taker)
2. **Unified Token Perspective**: All normalized to YES token
3. **Unified Direction**: All trades converted to BUY, negative amounts for sells
4. **Sorted by User**: Easy to analyze individual user trajectories

### Data Cleaning Process

#### Step 1: Filter Invalid Data (Same as quant)
```python
# Remove NaN prices and contract trades
df = df[~df['price'].isna()]
df = df[~df['taker'].str.lower().isin(contract_addresses)]
```

#### Step 2: Split Maker and Taker into Separate Records
```python
# Common fields for both sides
common = ['timestamp', 'datetime', 'block_number', 'transaction_hash',
          'market_id', 'price', 'token_amount', 'usd_amount']

# Maker record
maker_df = df[common + ['maker', 'maker_direction']].copy()
maker_df = maker_df.rename(columns={'maker': 'user', 'maker_direction': 'direction'})
maker_df['role'] = 'maker'

# Taker record
taker_df = df[common + ['taker', 'taker_direction']].copy()
taker_df = taker_df.rename(columns={'taker': 'user', 'taker_direction': 'direction'})
taker_df['role'] = 'taker'

# Concatenate
result = pd.concat([maker_df, taker_df])

# Original 1 trade → 2 records (expansion ratio ~2x)
```

#### Step 3: Unified Token Perspective (YES/token1)
```python
# If token2 trade, flip price (same as quant)
is_token2 = result['nonusdc_side'] == 'token2'
result.loc[is_token2, 'price'] = 1 - result.loc[is_token2, 'price']
```

#### Step 4: Unified Direction to BUY
```python
# Convert all SELL to BUY with negative token_amount
is_sell = result['direction'] == 'SELL'
result.loc[is_sell, 'token_amount'] = -result.loc[is_sell, 'token_amount']
result['direction'] = 'BUY'  # All records are now 'BUY'

# Example:
# Original SELL: direction='SELL', token_amount=100
# Unified BUY:  direction='BUY',  token_amount=-100
```

**Why?**
- **BUY** (token_amount > 0): User bought YES tokens, spent USDC
- **SELL** (token_amount < 0): User sold YES tokens, received USDC
- Single direction simplifies aggregation: `sum(token_amount)` = net position

#### Step 5: Sort by User and Time
```python
result = result.sort_values(['user', 'timestamp'])
```

### Schema
```
timestamp           int64      # Unix timestamp
datetime            string     # Human-readable datetime
block_number        int64      # Block number
transaction_hash    string     # Transaction hash
event_id            string     # Unique event identifier
market_id           string     # Market ID
condition_id        string     # Condition ID
user                string     # User address (was maker/taker)
role                string     # 'maker' or 'taker'
price               float64    # YES token price (unified)
token_amount        int64      # Token amount (negative if originally SELL)
usd_amount          int64      # USDC amount (wei)
```

### Key Features
- ✅ Each user appears in their own records
- ✅ Easy to track user trading history
- ✅ Unified BUY direction with signed amounts
- ✅ Sorted by user → timestamp for sequential analysis
- ✅ 2x expansion ratio (1 trade → 2 records)

### Use Cases
- **User Profiling**: Track individual user strategies
- **Wallet Analysis**: PnL calculation per user
- **Cohort Analysis**: User behavior segmentation
- **Position Tracking**: Sum token_amount = net position
- **Trading Pattern Detection**: Identify market makers, arbitrageurs
- **User Journey**: Sequential trade analysis

### Example: Calculate User Net Position
```python
import pandas as pd

users = pd.read_parquet('users.parquet')

# Net position per user per market
position = users.groupby(['user', 'market_id'])['token_amount'].sum()

# User with long position (bought more than sold)
positive_position = position[position > 0]

# User with short position (sold more than bought)
negative_position = position[position < 0]
```

---

## 🔄 Data Processing Pipeline

```
Polygon Blockchain
       ↓
   RPC Query (OrderFilled events)
       ↓
   Decode ABI
       ↓
orderfilled.parquet (31GB)
   ├─→ Link market_id via Gamma API
   ↓
trades.parquet (32GB)
   ├─→ Filter + Unified YES perspective
   ├─→ Keep maker/taker structure
   ↓
quant.parquet (21GB)
   │
   └─→ Split maker/taker + Unified BUY direction
       ↓
   users.parquet (23GB)
```

---

## 🆚 Comparison: quant vs users

| Feature | quant.parquet | users.parquet |
|---------|---------------|---------------|
| **Perspective** | Trade-centric | User-centric |
| **Records per trade** | 1 | 2 (maker + taker) |
| **Size** | 21GB | 23GB |
| **Token normalization** | ✅ YES (token1) | ✅ YES (token1) |
| **Direction** | Preserved (BUY/SELL) | Unified (BUY only) |
| **Maker/Taker** | Both preserved | Split into rows |
| **Sort order** | Original | User → Time |
| **Use case** | Trading algorithms | User behavior |

---

## 💡 Usage Examples

### Example 1: Load Quant Data for Backtesting
```python
import pandas as pd

quant = pd.read_parquet('quant.parquet')

# All trades in YES token perspective
# Price represents YES probability
market_data = quant[quant['market_id'] == 'specific_market']

# Calculate returns
market_data = market_data.sort_values('timestamp')
market_data['returns'] = market_data['price'].pct_change()
```

### Example 2: Analyze User Trading Patterns
```python
users = pd.read_parquet('users.parquet')

# Get one user's trading history
user_trades = users[users['user'] == '0x123...']

# Calculate net position (positive = long, negative = short)
net_position = user_trades.groupby('market_id')['token_amount'].sum()

# Identify strategy
if (user_trades['role'] == 'maker').mean() > 0.7:
    strategy = 'Market Maker'
elif net_position.abs().mean() < 1000:
    strategy = 'Scalper'
else:
    strategy = 'Position Trader'
```

---

## 📊 Data Quality

### Completeness
- ✅ All OrderFilled events since contract deployment
- ✅ No missing blocks (failed fetches are retried)
- ✅ All blockchain fields preserved

### Cleaning Rules
- ❌ Removed: Contract-to-contract trades
- ❌ Removed: Trades with NaN prices
- ✅ Kept: All user-to-user trades
- ✅ Kept: All fee information

### Token Normalization
- All `quant.parquet` and `users.parquet` prices represent **YES token probability**
- Original token2 prices are flipped: `price_yes = 1 - price_no`
- Consistent for time-series and cross-market analysis

---

## 🔗 Relationships

```
markets.parquet
    ↓ (market_id)
trades.parquet
    ↓ (filter + normalize)
quant.parquet (trade-level)
users.parquet (user-level)
```

---

## ⚖️ License

MIT License - Free for commercial and research use

---

## 📞 Questions?

- GitHub Issues: [polymarket-data](https://github.com/SII-WANGZJ/polymarket-data/issues)
- Dataset: [HuggingFace](https://huggingface.co/datasets/SII-WANGZJ/Polymarket_data)
