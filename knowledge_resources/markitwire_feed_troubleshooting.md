# MarkitWire Feed Troubleshooting

## Overview
This guide covers common issues and resolution steps for MarkitWire trade feed problems.

## Common Issues

### 1. Feed Eligibility Not Met
**Symptoms:** Trade doesn't feed to MarkitWire
**Resolution Steps:**
- Verify book2 had been resolved to type "MarkitWire"
- If book2 has not resolved correctly, check eligibility object's IneligibilityReasons
- If book2 was resolved correctly check if there are any block events
- Make sure downstream trade passes validation

### 2. Trade Validation Failures
**Symptoms:** Feed rejected with validation errors
**Resolution Steps:**
```python
deal = ro(dealName)
fs = deal.FeedState("MarkitWire")
dt = fs._DownstreamTrades()[0]
dt.validate() #investigate validation failures
```

### 3. Downstream Processing Errors
**Symptoms:** Feed sent but processing fails
**Resolution Steps:**
- Check block events on downstream state
- Compare values in Athena to what MW is expecting

## Monitoring Commands
```python
# Check feed status
deal = ro(dealName)
fs = deal.FeedState("MarkitWire")
fs.FeedStatus()
```

## Escalation Path
If feeds continue to fail after following these steps:
Contact ATRS MarkitWire Support with trade details and repro
