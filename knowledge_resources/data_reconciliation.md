# Data Reconciliation Issue Resolution

## Overview
This runbook covers procedures for investigating and resolving data reconciliation discrepancies.

## Common Reconciliation Issues

### 1. Position Report Discrepancies
**Symptoms:** 
- Position totals don't match between systems
- Missing trades in position calculations
- Incorrect mark-to-market valuations

**Investigation Steps:**
1. **Identify Scope:**
   - Which positions are affected?
   - What time period shows discrepancies?
   - Which systems are involved?

2. **Data Source Analysis:**
   ```sql
   -- Check trade booking completeness
   SELECT booking_date, COUNT(*) as trade_count 
   FROM trades 
   WHERE booking_date = CURRENT_DATE
   GROUP BY booking_date;
   
   -- Verify position calculations
   SELECT portfolio, SUM(quantity * price) as total_value
   FROM positions 
   WHERE as_of_date = CURRENT_DATE
   GROUP BY portfolio;
   ```

3. **Reconciliation Validation:**
   - Run data validation rules manually
   - Compare record counts between systems
   - Check for data type mismatches or nulls
   - Verify timestamp consistency across systems

### 2. Downstream Data Issues
**Symptoms:**
- Data not appearing in target systems
- Transformation errors in processing
- Data quality validation failures

**Resolution Process:**
1. **Pipeline Health Check:**
   - Check STP job execution logs
   - Verify data transformation logic
   - Validate mapping rules and custom logic

2. **Data Quality Validation:**
   ```bash
   # Check file arrival and processing
   ls -la /data/incoming/{YYYYMMDD}/
   
   # Validate data format
   head -n 10 /data/incoming/trades_20240101.csv
   
   # Check processing status
   SELECT job_name, status, start_time, end_time 
   FROM etl_job_history 
   WHERE run_date = CURRENT_DATE;
   ```

### 3. Regulatory Reporting Discrepancies
**Critical Actions:**
- Identify affected regulatory reports
- Calculate potential exposure and impact
- Coordinate with Compliance team immediately
- Document all investigation steps

## Data Correction Procedures

### 1. Trade Data Corrections
- Verify authorization for data changes
- Document business justification
- Apply corrections in designated maintenance window
- Validate corrections across all downstream systems

### 2. Reprocessing Requirements
- Identify data dependencies and downstream impacts
- Schedule reprocessing during low-activity periods
- Validate data consistency after reprocessing
- Update reconciliation exceptions if valid

## Quality Assurance
- Run full reconciliation after corrections
- Compare before/after data states
- Validate with business users
- Update monitoring to detect similar issues

## Documentation Requirements
- Record all investigation steps and findings
- Document root cause analysis
- Update data lineage documentation
- Create preventive measures for future occurrences
