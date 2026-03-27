"""
Utility functions for GenoPHI package.

This module provides general utility and validation functions to ensure data
integrity and compatibility across different modeling tasks.
"""

import os
import logging
import pandas as pd
import numpy as np


def validate_phenotype_task_type(y, task_type, phenotype_column='interaction'):
    """
    Validates that the phenotype data type matches the specified task type.

    This function checks whether the phenotype data is appropriate for the
    specified task (classification or regression) and raises an error if there's
    a mismatch. This prevents users from accidentally running classification
    on continuous data or vice versa.

    Args:
        y (pd.Series): The phenotype/target variable to validate.
        task_type (str): The task type - must be 'classification' or 'regression'.
        phenotype_column (str): Name of the phenotype column for error messages.
            Default is 'interaction'.

    Raises:
        ValueError: If task_type is invalid or if the phenotype data type doesn't
            match the task type.

    Examples:
        >>> import pandas as pd
        >>> # Continuous data - should work with regression
        >>> y = pd.Series([1.5, 2.3, 4.7, 3.2])
        >>> validate_phenotype_task_type(y, 'regression', 'score')

        >>> # Binary data - should work with classification
        >>> y = pd.Series([0, 1, 0, 1, 1])
        >>> validate_phenotype_task_type(y, 'classification', 'label')

        >>> # Continuous data with classification - should raise error
        >>> y = pd.Series([1.5, 2.3, 4.7, 3.2])
        >>> validate_phenotype_task_type(y, 'classification', 'score')
        Traceback (most recent call last):
            ...
        ValueError: Task type is set to 'classification', but the phenotype column...
    """
    # Validate task_type parameter
    if task_type not in ['classification', 'regression']:
        raise ValueError(
            f"Invalid task_type '{task_type}'. Must be 'classification' or 'regression'."
        )

    # Get basic statistics about the data
    unique_values = y.nunique()

    # Detect data type characteristics
    is_float_type = pd.api.types.is_float_dtype(y)
    is_integer_type = pd.api.types.is_integer_dtype(y)
    is_numeric = pd.api.types.is_numeric_dtype(y)

    if task_type == 'classification':
        # Classification should have discrete categorical values
        # Check 1: If data is float type with many unique values, likely continuous
        if is_float_type and unique_values > 10:
            raise ValueError(
                f"Task type is set to 'classification', but the phenotype column '{phenotype_column}' "
                f"contains continuous data with {unique_values} unique values.\n"
                f"Data type: {y.dtype}\n"
                f"Sample values: {y.head().tolist()}\n\n"
                f"For continuous/numeric phenotype data, please use:\n"
                f"  --task_type regression\n\n"
                f"If this is truly categorical data, please encode it as integers (0, 1, 2, etc.)."
            )

        # Check 2: If float values have decimal places, likely continuous
        if is_float_type:
            # Sample first 100 non-null values to check for decimals
            sample_values = y.dropna().head(100)
            if len(sample_values) > 0:
                has_decimals = any(val % 1 != 0 for val in sample_values if pd.notna(val))
                if has_decimals:
                    raise ValueError(
                        f"Task type is set to 'classification', but the phenotype column '{phenotype_column}' "
                        f"contains float values with decimal places.\n"
                        f"Sample values: {y.dropna().head(5).tolist()}\n\n"
                        f"Classification requires discrete categories (e.g., 0, 1, 2).\n"
                        f"For continuous/numeric phenotype data, please use:\n"
                        f"  --task_type regression"
                    )

        # Check 3: Warn if there are many unique values even for integer data
        if is_integer_type and unique_values > 20:
            logging.warning(
                f"Phenotype column '{phenotype_column}' has {unique_values} unique integer values "
                f"for classification. This is unusual for categorical data. "
                f"Please verify this is the correct task type."
            )

        logging.info(
            f"Validation passed: Phenotype data is suitable for classification "
            f"(unique classes: {unique_values}, dtype: {y.dtype})"
        )

    elif task_type == 'regression':
        # For regression, we expect continuous data or many unique values
        # Provide a warning if data looks categorical/binary
        if is_numeric and unique_values <= 2:
            unique_vals_list = sorted(y.unique())
            logging.warning(
                f"Task type is set to 'regression', but the phenotype column '{phenotype_column}' "
                f"appears to contain binary/categorical data with only {unique_values} unique value(s): {unique_vals_list}.\n"
                f"If this is a classification problem (predicting categories), consider using:\n"
                f"  --task_type classification"
            )

        # Warn if data looks like multi-class categorical
        if is_integer_type and 3 <= unique_values <= 10:
            logging.warning(
                f"Task type is set to 'regression', but the phenotype column '{phenotype_column}' "
                f"contains {unique_values} unique integer values, which may indicate categorical data. "
                f"If predicting categories, consider using --task_type classification."
            )

        logging.info(
            f"Validation passed: Phenotype data is suitable for regression "
            f"(unique values: {unique_values}, dtype: {y.dtype})"
        )


def validate_file(path, name):
    """
    Validate that a file exists and is accessible.

    Args:
        path (str): Path to the file to validate.
        name (str): Descriptive name of the file for error messages.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the path is not a file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")
    if not os.path.isfile(path):
        raise ValueError(f"{name} is not a file: {path}")


def validate_directory(path, name, create=False):
    """
    Validate or create a directory.

    Args:
        path (str): Path to the directory to validate.
        name (str): Descriptive name of the directory for error messages.
        create (bool): If True, create the directory if it doesn't exist.
            Default is False.

    Raises:
        FileNotFoundError: If the directory does not exist and create is False.
        ValueError: If the path exists but is not a directory.
    """
    if create:
        os.makedirs(path, exist_ok=True)
    elif not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")
    elif not os.path.isdir(path):
        raise ValueError(f"{name} is not a directory: {path}")
