# MarkitWire Feed Troubleshooting

## Overview
This guide covers common issues and resolution steps for MarkitWire trade feed problems.

## Common Issues

### 1. Feed Eligibility Not Met
**Symptoms:** Trade appears in booking system but doesn't feed to MarkitWire
**Resolution Steps:**
- Verify trade meets minimum notional requirements
- Check product type is supported for MarkitWire feeds
- Validate counterparty is approved for feeds
- Review trade maturity date requirements

### 2. Trade Validation Failures
**Symptoms:** Feed rejected with validation errors
**Resolution Steps:**
- Check all required fields are populated
- Validate trade economics (rates, spreads, etc.)
- Verify regulatory classifications
- Review trade booking validation results

### 3. Downstream Processing Errors
**Symptoms:** Feed sent but processing fails
**Resolution Steps:**
- Check trade status in downstream systems
- Verify feed acknowledgment was received
- Review any error responses from MarkitWire
- Check connectivity to MarkitWire gateway

## Monitoring Commands
```bash
# Check feed status
SELECT * FROM trade_feeds WHERE trade_id = '<trade_id>';

# Review validation results  
SELECT * FROM feed_validations WHERE trade_id = '<trade_id>';

# Check MarkitWire connectivity
curl -H "Authorization: Bearer $TOKEN" https://api.markitwire.com/health
```

## Escalation Path
If feeds continue to fail after following these steps:
1. Contact MarkitWire support with trade details
2. Escalate to Trade Operations team
3. Engage Infrastructure team for connectivity issues
