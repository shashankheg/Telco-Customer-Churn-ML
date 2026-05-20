
from typing import Tuple, List
import great_expectations as gx

def validate_PredMaintdata(df) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Predictive Maintenance dataset using Great Expectations.
    """
    print("🔍 Starting data validation with Great Expectations...")

    # === SETUP: Create in-memory context and validator ===
    context = gx.get_context()
    data_source = context.data_sources.add_pandas("my_datasource")
    data_asset = data_source.add_dataframe_asset("my_dataframe")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("my_batch")

    # === BUILD EXPECTATION SUITE ===
    print("   📋 Validating schema and required columns...")
    suite = context.suites.add(gx.ExpectationSuite(name="predictive_maintenance_suite"))

    # Schema validation
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="Type"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="Failure Type"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="Rotational speed [rpm]"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="Torque [Nm]"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="Air temperature [K]"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="Process temperature [K]"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="Target"))

    # Null checks
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="Failure Type"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="Target"))

    # Business logic validation
    print("   💼 Validating business logic constraints...")
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(
        column="Type", value_set=["L", "M", "H"]
    ))

    # === RUN VALIDATION ===
    validation_definition = context.validation_definitions.add(
        gx.ValidationDefinition(
            name="my_validation_definition",
            data=batch_definition,
            suite=suite
        )
    )

    results = validation_definition.run(batch_parameters={"dataframe": df})

    print(type(results))

    # === PROCESS RESULTS ===
    failed_expectations = []
    for r in results.results:
        if not r.success:
            failed_expectations.append(r.expectation_config.type)

    total_checks = len(results.results)
    passed_checks = sum(1 for r in results.results if r.success)
    failed_checks = total_checks - passed_checks

    if results.success:
        print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"❌ Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        print(f"   Failed expectations: {failed_expectations}")

    return results.success, failed_expectations