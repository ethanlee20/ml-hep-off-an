import pandas as pd


def bin_data(df, names_columns_to_bin, number_of_bins):
    for column_name in names_columns_to_bin:
        df[column_name + "_bin"] = pd.cut(
            df[column_name],
            bins=number_of_bins,
            labels=False,
        )
    return df


def averages_per_bin(df, names_bin_columns):
    return df.groupby(names_bin_columns).mean()


def bin_avgs(df, names_columns_to_bin, number_of_bins):
    df = bin_data(df, names_columns_to_bin, number_of_bins)

    names_bin_columns = [
        name + "_bin" for name in names_columns_to_bin
    ]
    df_binned_avgs = averages_per_bin(df, names_bin_columns)
    df_binned_avgs = df_binned_avgs.reset_index()

    return df_binned_avgs


def generate_image_dataframe(
    data_df,
    names_columns_to_bin,
    number_of_bins,
    name_q_squared_column,
):
    binned_avgs = bin_avgs(
        data_df, names_columns_to_bin, number_of_bins
    )
    names_bin_columns = [
        name + "_bin" for name in names_columns_to_bin
    ]
    image_df = binned_avgs[
        names_bin_columns + [name_q_squared_column]
    ]
    return image_df
