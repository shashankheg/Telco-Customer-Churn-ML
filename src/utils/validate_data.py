# import great_expectations as gx
# from typing import Tuple, List


# def validate_telco_data(df) -> Tuple[bool, List[str]]:
#     """
#     Comprehensive data validation for Telco Customer Churn dataset using Great Expectations.
    
#     This function implements critical data quality checks that must pass before model training.
#     It validates data integrity, business logic constraints, and statistical properties
#     that the ML model expects.
    
#     """
#     print("🔍 Starting data validation with Great Expectations...")

#     context = gx.get_context()
#     data_source = context.data_sources.add_pandas("my_datasource")
#     data_asset = data_source.add_dataframe_asset("my_dataframe")
#     batch_definition = data_asset.add_batch_definition_whole_dataframe("my_batch")
    
#     # Convert pandas DataFrame to Great Expectations Dataset
#     # ge_df = ge.dataset.PandasDataset(df)
    
#     # === SCHEMA VALIDATION - ESSENTIAL COLUMNS ===
#     print("   📋 Validating schema and required columns...")
    
#     # Customer identifier must exist (required for business operations) 
#     # 
#     #  
#     suite = context.suites.add(gx.ExpectationSuite(name="Customer Churn dataset"))

#     # Schema validation
#     suite.add_expectation(gx.expectations.ExpectColumnToExist(column="customerID"))
#     suite.add_expectation(gx.expectations.ExpectColumnToExist(column="gender"))
#     suite.add_expectation(gx.expectations.ExpectColumnToExist(column="Partner"))
#     suite.add_expectation(gx.expectations.ExpectColumnToExist(column="Dependents"))
#     suite.add_expectation(gx.expectations.ExpectColumnToExist(column="PhoneService"))
#     suite.add_expectation(gx.expectations.ExpectColumnToExist(column="InternetService"))
#     suite.add_expectation(gx.expectations.ExpectColumnToExist(column="Contract"))
#     suite.add_expectation(gx.expectations.ExpectColumnToExist(column="tenure"))
#     suite.add_expectation(gx.expectations.ExpectColumnToExist(column="MonthlyCharges"))
#     suite.add_expectation(gx.expectations.ExpectColumnToExist(column="TotalCharges"))

#     # Null checks
#     suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="Target"))
     
#     # === BUSINESS LOGIC VALIDATION ===
#     print("   💼 Validating business logic constraints...")
    
#     # Gender must be one of expected values (data integrity)

#     suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(
#         column="gender", value_set=["Male", "Female"]
#     ))
      
#     # Yes/No fields must have valid values
#     suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(
#         column="Partner", value_set=["Yes", "No"]
#     ))
#     suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(
#         column="Dependents", value_set=["Yes", "No"]
#     ))
#     suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(
#         column="PhoneService", value_set=["Yes", "No"]
#     ))
    
#        # Contract types must be valid (business constraint)
#     suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(
#         column="Contract", 
#         value_set=["Month-to-month", "One year", "Two year"]
#     ))
    
#     # Internet service types (business constraint)
#     suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(
#         column="InternetService",
#         value_set=["DSL", "Fiber optic", "No"]
#     ))
    


#     # === NUMERIC RANGE VALIDATION ===
#     print("   📊 Validating numeric ranges and business constraints...")
    
#     # Tenure must be non-negative (business logic - can't have negative tenure)
#     suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
#         column="tenure", min_value=0
#     ))
#     # Monthly charges must be positive (business logic - no free service)
#     suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
#         column="MonthlyCharges", min_value=0
#     ))
    
#     # Total charges should be non-negative (business logic)
#     suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
#         column="TotalCharges", min_value=0
#     ))
    
#     # === STATISTICAL VALIDATION ===
#     print("   📈 Validating statistical properties...")
    
#     # Tenure should be reasonable (max ~10 years = 120 months for telecom)
#     # Monthly charges should be within reasonable business range
#     # No missing values in critical numeric features  
#     suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="tenure"))
#     suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="MonthlyCharges"))
    
#     # === DATA CONSISTENCY CHECKS ===
#     print("   🔗 Validating data consistency...")
    
#     # Total charges should generally be >= Monthly charges (except for very new customers)
#     # This is a business logic check to catch data entry errors
#     suite.add_expectation(gx.expectations.ExpectColumnPairValuesAToBeGreaterThanB(
    
#         column_A="TotalCharges",
#         column_B="MonthlyCharges",
#         or_equal=True,
#         mostly=0.95  # Allow 5% exceptions for edge cases
#     ))    
#     # === RUN VALIDATION SUITE ===
#     print("   ⚙️  Running complete validation suite...")

#     validation_definition = context.validation_definitions.add(
#         gx.ValidationDefinition(
#             name="my_validation_definition",
#             data=batch_definition,
#             suite=suite
#         )
#     )
#     results = validation_definition.run(batch_parameters={"dataframe": df})
#     print(results.keys())
#     print(results.results)

    
#     # === PROCESS RESULTS ===
#     # Extract failed expectations for detailed error reporting
#   # --- Extract failed expectations ---
#     failed = []
#     for r in results.get("results", []):
#         if not r.get("success", True):
#             cfg = r["expectation_config"]
#             etype = getattr(cfg, "expectation_type", None)
#             kw = cfg.kwargs

#             col = kw.get("column") or kw.get("column_A") or kw.get("column_B")
#             msg = etype or f"Expectation failed"
#             failed.append(f"{msg} on '{col}' with {kw}")

# # --- Summary ---
#     res = results.get("results", [])
#     total = len(res)
#     passed = sum(r.get("success", False) for r in res)
#     failed_count = total - passed

#     if results.get("success", False):
#         print(f"✅ Passed {passed}/{total}")
#     else:
#         print(f"❌ Failed {failed_count}/{total}")
#         print("   Failed expectations:", failed)

#     return results.get("success", False), failed

import great_expectations as gx
from typing import Tuple, List

def validate_telco_data(df) -> Tuple[bool, List[str]]:
    context = gx.get_context()
    batch_def = (
        context.data_sources.add_pandas("pandas_ds")
        .add_dataframe_asset("df_asset")
        .add_batch_definition_whole_dataframe("batch")
    )

    suite = context.suites.add(gx.ExpectationSuite(name="telco_churn_suite"))

    for col in ["customerID", "gender", "Partner", "Dependents", "PhoneService",
                "InternetService", "Contract", "tenure", "MonthlyCharges", "TotalCharges"]:
        suite.add_expectation(gx.expectations.ExpectColumnToExist(column=col))

    for col in ["Churn", "tenure", "MonthlyCharges"]:
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=col))

    categorical = {
        "gender":          ["Male", "Female"],
        "Partner":         ["Yes", "No"],
        "Dependents":      ["Yes", "No"],
        "PhoneService":    ["Yes", "No"],
        "Contract":        ["Month-to-month", "One year", "Two year"],
        "InternetService": ["DSL", "Fiber optic", "No"],
    }
    for col, values in categorical.items():
        suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(
            column=col, value_set=values
        ))

    for col in ["tenure", "MonthlyCharges"]:
        suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
            column=col, min_value=0
        ))

    results = context.validation_definitions.add(
        gx.ValidationDefinition(name="telco_validation", data=batch_def, suite=suite)
    ).run(batch_parameters={"dataframe": df})

    seen = set()
    failed = []
    for r in results.get("results", []):
        if not r.get("success", True):
            cfg = r["expectation_config"]
            etype = (
                getattr(cfg, "type", None)
                or getattr(cfg, "expectation_type", None)
                or type(cfg).__name__
            )
            col = cfg.kwargs.get("column") or cfg.kwargs.get("column_A") or "unknown"
            key = f"{etype} on '{col}'"
            if key not in seen:
                seen.add(key)
                failed.append(key)

    total = len(results.get("results", []))
    passed = total - len(failed)
    status = "✅" if results.get("success") else "❌"
    print(f"{status} {passed}/{total} passed" + (f" | Failed: {failed}" if failed else ""))

    return results.get("success", False), failed