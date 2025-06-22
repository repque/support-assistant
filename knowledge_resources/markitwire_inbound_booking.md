# MarkitWire Inbound Trade Booking Issues

## Overview
This guide covers common issues with **inbound** MarkitWire proxy trade booking (not outbound feeds).
Use this for issues where MarkitWire trades are not getting booked into Athena.

## Common Issues

### 1. Inbound Proxy Trade Booking Failure
**Symptoms:** MarkitWire proxy trade doesn't get booked in Athena from inbound messages
**Resolution Steps:**
- Check logs of Notification Listener to make sure inbound message had been sent
- Check hydra queues - make sure booking request is not stuck within .taken folder
- Verify UpstreamStateBuilder log processed inbound request correctly and created booking request
- Check STP booker's log for any errors
