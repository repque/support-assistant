# Trade Feed Troubleshooting

## Overview
This guide covers common issues and resolution steps for trade feed problems across all feed types.

## Common Issues

### 1. Feed Eligibility Not Met
**Symptoms:** Trade doesn't feed to target system
**Resolution Steps:**
- Verify book2 had been resolved to correct feed type
- If book2 has not resolved correctly, check eligibility object's IneligibilityReasons
- If book2 was resolved correctly check if there are any block events
- Make sure downstream trade passes validation

### 2. Trade Validation Failures
**Symptoms:** Feed rejected with validation errors
**Resolution Steps:**
```python
deal = ro(dealName)
fs = deal.FeedState(feedType)  # Use specific feed name like "DCPP", "Bloomberg", "Reuters" etc.
dt = fs._DownstreamTrades()[0]
dt.validate() #investigate validation failures
```

### 3. Downstream Processing Errors
**Symptoms:** Feed sent but processing fails
**Resolution Steps:**
- Check block events on downstream state
- Compare values in Athena to what target system is expecting

## Monitoring Commands
```python
# Check feed status
deal = ro(dealName)
fs = deal.FeedState(feedType)  # Use specific feed name like "DCPP", "Bloomberg", "Reuters" etc.
fs.FeedStatus()
```

## Escalation Path
If feeds continue to fail after following these steps:
Contact ATRS support team with trade details and reproduction steps
