# br_ppo_crypto_v15

Paper-trading repository for `br_ppo_crypto_v15`, based on the uploaded `V8_CRYPTO_MANDATORY_FREQAI_SHARPE_MAX_V84` artifacts.

## What this repo does

- Loads the V15/V8 metadata from `models/v8_crypto_mandatory_freqai_sharpe_max_v84_metadata.json`.
- Loads the PPO ensemble artifacts from `models/v8_crypto_mandatory_freqai_sharpe_max_v84_member_*.zip`.
- Uses PPO inference when `ALLOCATION_MODE=ppo`.
- Fails closed by default when `REQUIRE_PPO=true`; it will not silently trade `DEFAULT_ACTION` if PPO inference fails.
- Writes dashboard-compatible logs under `logs/`.
- Submits Alpaca paper orders only when `SUBMIT_ORDERS=true` and the rebalance gate allows trading.

## Required GitHub Actions secrets

Add these in `Settings -> Secrets and variables -> Actions -> Secrets`:

```text
ALPACA_CRYPTO_V15_KEY_ID
ALPACA_CRYPTO_V15_SECRET_KEY
```

## Recommended GitHub Actions variables

Add these in `Settings -> Secrets and variables -> Actions -> Variables`:

```text
ALLOCATION_MODE=ppo
REQUIRE_PPO=true
INCLUDE_PRIMARY_MODEL=false
DEFAULT_ACTION=crypto10_freqai15_llm25_ichi25_bil25
SUBMIT_ORDERS=false
CANCEL_OPEN_ORDERS=true
FORCE_REBALANCE=false
REBALANCE_EVERY_DAYS=10
MIN_ORDER_NOTIONAL=25
```

Start with `SUBMIT_ORDERS=false`. After checking `logs/orders/latest_planned_orders.csv`, set `SUBMIT_ORDERS=true`.

For a manual first live paper run, temporarily set:

```text
FORCE_REBALANCE=true
```

After the run, set it back to:

```text
FORCE_REBALANCE=false
```

## Verification after workflow run

Open `logs/decisions/latest_decision.csv` and confirm:

```text
action_source = ppo_ensemble_model
fallback_used = false
require_ppo = true
account_status = connected
```

If `orders_allowed=false`, the rebalance gate blocked trading. Use `FORCE_REBALANCE=true` for a manual override.

## Add to QSentia site

Copy the contents of `model_registry_entry.yaml` into `Base_Model_BR_PPO/models.yaml` under `models:`.
