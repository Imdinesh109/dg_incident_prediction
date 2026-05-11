
# Dangerous Goods Incident Prediction (CO-004)

## Mission Objective
Predict the likelihood of dangerous goods (DG) shipment incidents to prioritize inspections, improve operational safety, and reduce risks such as aircraft grounding, regulatory fines, and safety hazards.

## Target Variable
incident_flag
- 0 = No Incident
- 1 = Safety Incident / Regulatory Violation

## Dataset Overview
Total Records: 21,000

### Dataset Splits
- Training: 14,000 records (March 2025 - November 2025)
- Validation: 3,500 records (December 2025)
- Test: 3,500 records (January 2026 - February 2026)

### Class Distribution
- Approx. 82% No Incident
- Approx. 18% Incident

## Feature Descriptions

| Feature | Description | Type |
|---|---|---|
| record_id | Unique shipment identifier | String |
| timestamp | Shipment timestamp | Datetime |
| shc_code | Special handling code | Categorical |
| origin_destination | Shipment route profile | Categorical |
| dg_class | Dangerous goods regulatory class | Categorical |
| packaging_type | Packaging structure type | Categorical |
| handling_error_count | Handling mistakes during operations | Integer |
| previous_incident_count | Historical incidents linked to shipper | Integer |
| cargo_weight_kg | Cargo weight in kilograms | Float |
| temperature_celsius | Shipment temperature exposure | Float |
| humidity_percentage | Environmental humidity | Float |
| weather_condition | Weather during shipment | Categorical |
| safety_staff_count | Operational oversight staffing | Integer |
| shipper_id | Shipper identifier | Categorical |
| doc_audit_result | Documentation audit result | Binary |
| incident_flag | Prediction target | Binary |

## Embedded ML Learning Patterns
1. Handling errors greater than 7 sharply increase incident probability.
2. OSL-JED routes combined with DG classes 2.x and 3 have elevated climate shock risks.
3. Documentation audit failures strongly increase incident probability.
4. CAO shipments with weak packaging structures elevate risk.
5. High historical shipper incidents increase baseline risk.

## Recommended ML Use Cases
- Classification Models
- Risk Scoring
- Inspection Prioritization
- Operational Safety Dashboards
- Explainable AI Risk Analysis

