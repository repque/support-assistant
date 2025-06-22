# Feed Framework Troubleshooting

## Overview
This guide covers common issues and resolution steps for MarkitWire trade feed problems.

## Common Issues

### 1. Getting downstream trade for a deal
**Symptoms:** Need to get an instance of downstream trade for a given feed type
**Resolution Steps:**
```python
d = ro(dealName)
fs = d.FeedState(feedType) # feedType can be "MarkitWire" or "XODS" etc.
dt = fs._DownstreamTrades()[0]
```

### 2. Check FeedStatus for a given feed type
**Symptoms:** Feed status checking 
**Resolution Steps:**
```python
deal = ro(dealName)
fs = deal.FeedState(feedType)  # Use specific feed name like "DCPP", "Bloomberg", "Reuters" etc.
fs.FeedStatus()
```

### 3. Check downstream events for a given feed type
**Symptoms:** Need to see the list of downstream events
**Resolution Steps:**
```python
deal = ro(dealName)
ds = deal.DownstreamState(feedType)  # Use specific feed name like "DCPP", "Bloomberg", "Reuters" etc.
ds.evInfo() # this should print out downstream events, such as update events and block events
```