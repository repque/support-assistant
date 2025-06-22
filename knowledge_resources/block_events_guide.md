# Block Events Investigation Guide

## Overview
Block events prevent trades from feeding to downstream systems. This guide explains how to investigate and resolve block events.

## Checking Block Events

### 1. View Block Events on Feed State
```python
# Get the feed state
deal = ro(dealName)
fs = deal.FeedState("MarkitWire")

# Check for block events
block_events = fs._BlockEvents()
for event in block_events:
    print(f"Block Type: {event.Type}")
    print(f"Reason: {event.Reason}")
    print(f"Timestamp: {event.Timestamp}")
```

### 2. Common Block Event Types
- **ValidationBlock**: Trade failed validation rules
- **EligibilityBlock**: Trade doesn't meet eligibility criteria
- **SystemBlock**: System-level issues preventing feed
- **ManualBlock**: Manually blocked by operations

### 3. Resolution Steps
1. Identify the block type from `_BlockEvents()`
2. Address the underlying cause
3. Clear the block if appropriate
4. Revalidate and resend the trade

## Clearing Block Events
```python
# Clear specific block event
fs.ClearBlockEvent(event_id)

# Force resend after clearing blocks
fs.Resend()
```