
# Claim Submit File Analysis Report

## Key Observations

1. **Data Completeness**:
   - The majority of columns have high non-null counts, indicating well-populated datasets.
   - Columns like `claim_number`, `vendorname`, and `clearinghouse_received_date` are fully populated, suggesting these are essential for every record

2. **High Cardinality**:
   - Columns like `claim_number`, `vendorname`, and `cycle_id` exhibit high cardinality, suggesting diverse entries. This variability is expected given the transactional nature of the data.

3. **Data Gaps**:
   - Significant data gaps are present in columns like `facility_provider_address_*` and `secondary_payer_pay_type`, with non-null counts dropping below 10% in some cases. These columns can be ignored as they can be inferred from other columns

