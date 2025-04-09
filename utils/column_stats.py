import pandas as pd
import numpy as np
import json


def _generate_base_stats(series):
    return {
        "data_type": str(series.dtype),
        "rows": int(len(series)),
        "distinct_values": int(series.nunique()),
        "missing_values": int(series.isnull().sum()),
    }


def _get_mode_and_freq(series):
    if series.isnull().all():
        return None, 0

    mode_result = series.mode()
    if not mode_result.empty:
        most_frequent_value = mode_result.iloc[0]
        frequency = int(series.value_counts().get(most_frequent_value, 0))
        return most_frequent_value, frequency
    else:
        return None, 0


def _safe_float(value):
    try:
        if value is not None and pd.notnull(value):
            return float(value)
    except (TypeError, ValueError):
        pass
    return None


def safe_str_date(value):
    if value is not None and pd.notnull(value):
        return value.isoformat()
    return None


def _generate_numeric_stats(series):
    base_stats = _generate_base_stats(series)
    desc = series.describe()
    
    numeric_stats = {
        "mean": _safe_float(desc.get('mean')),
        "standard_deviation": _safe_float(desc.get('std')),
        "minimum": _safe_float(desc.get('min')),
        "25th_percentile": _safe_float(desc.get('25%')),
        "median": _safe_float(desc.get('50%')),
        "75th_percentile": _safe_float(desc.get('75%')),
        "maximum": _safe_float(desc.get('max'))
    }
    
    base_stats.update(numeric_stats)
    return base_stats


def _generate_common_categorical_stats(series):
    base_stats = _generate_base_stats(series)
    most_frequent_value, frequency = _get_mode_and_freq(series)
    categorical_stats = {
        "most_frequent_value": most_frequent_value,
        "frequency": int(frequency)
    }
    base_stats.update(categorical_stats)
    return base_stats


def _generate_datetime_stats(series):
    base_stats = _generate_base_stats(series)
    most_frequent_value, frequency = _get_mode_and_freq(series)
    
    min_date = safe_str_date(series.min())
    max_date = safe_str_date(series.max())
    m_f_value = safe_str_date(most_frequent_value)
    
    datetime_stats = {
        "minimum_date": min_date,
        "maximum_date": max_date,
        "most_frequent_value": m_f_value,
        "frequency": int(frequency)
    }
    
    base_stats.update(datetime_stats)
    return base_stats
    

def _generate_unsupported_stats(series):
    base_stats = _generate_base_stats(series)
    base_stats["note"] = "Stats generation not fully supported for this type."
    return base_stats


STAT_GENERATORS = {
    'numeric': lambda s: _generate_numeric_stats(s),
    'bool': lambda s: _generate_common_categorical_stats(s),
    'categorical': lambda s: _generate_common_categorical_stats(s),
    'object': lambda s: _generate_common_categorical_stats(s),
    'datetime': lambda s: _generate_datetime_stats(s),
}


def _get_stat_generator(series):
    if pd.api.types.is_bool_dtype(series) or series.dtype == 'boolean':
        return STAT_GENERATORS['bool']
    elif pd.api.types.is_numeric_dtype(series):
        return STAT_GENERATORS['numeric']
    elif isinstance(series, pd.CategoricalDtype):
        return STAT_GENERATORS['categorical']
    elif series.dtype == "object":
        return STAT_GENERATORS['object']
    elif pd.api.types.is_datetime64_any_dtype(series):
        return STAT_GENERATORS['datetime']
    else:
        return _generate_unsupported_stats


def generate_column_stats(df):
    """
    Generates descriptive statistics for each column in a DataFrame,
    formatted based on the column's data type.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        dict: A dictionary of dictionaries, where each key is column's name 
              and each dictionary contains column's stats
    """
    stats = {}
    for column in df.columns:
        series = df[column]
        stat_generator = _get_stat_generator(series)
        stats[column] = {"stats": stat_generator(series)}
        
    return stats


def main():
    exp = "mlp-nsplits-10"
    df_paper  = pd.read_parquet(f"data-paper/{exp}.parquet.gzip")
    stats_dict = generate_column_stats(df=df_paper)
    
    with open(f"misc/{exp}-stats.json", "w") as f:
        json.dump(stats_dict, f, indent=4, default=str)


if __name__ == "__main__":
    main()