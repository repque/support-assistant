# System Outage Investigation Procedures

## Immediate Response Checklist

### 1. Confirm Outage Scope
- [ ] Check system health dashboards
- [ ] Review recent alerts and notifications  
- [ ] Test service endpoints from multiple locations
- [ ] Verify impact scope (single service vs. multiple systems)

### 2. Initial Diagnostics
- [ ] Check infrastructure monitoring (CPU, memory, disk, network)
- [ ] Review load balancer status and routing
- [ ] Verify database connectivity and performance
- [ ] Check external dependencies and third-party services

## Investigation Steps

### 1. Recent Changes Analysis
- Review deployments in last 24 hours
- Check configuration changes
- Validate recent database migrations
- Examine infrastructure modifications

### 2. Log Analysis
**Application Logs:**
```bash
# Check for errors in last hour
grep -i error /var/log/app/*.log | tail -100

# Look for specific service issues
journalctl -u trade-booking-service --since "1 hour ago"
```

**Infrastructure Logs:**
- Load balancer access logs
- Database connection logs
- Network connectivity logs
- Security audit logs

### 3. Dependency Validation
- Test all upstream service dependencies
- Verify API endpoints are responding
- Check database cluster health
- Validate message queue status

## Resolution Framework

### Critical Service Recovery
1. **Immediate:** Implement service failover if available
2. **Short-term:** Apply emergency fixes or rollbacks
3. **Long-term:** Address root cause and improve monitoring

### Communication Protocol
- Notify stakeholders within 15 minutes
- Provide hourly updates during resolution
- Send final incident report within 24 hours

## Post-Incident Actions
- Conduct blameless post-mortem
- Update runbooks based on lessons learned
- Enhance monitoring to detect similar issues
- Review and test disaster recovery procedures
