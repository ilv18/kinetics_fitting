def main():
    import kinetics_fitting.data as data_mod
    from kinetics_fitting.data import dataExtractor, wavelengthMerger, waveExtractor
    from kinetics_fitting.fitting import fit_reaction_segments, compute_inflection
    from kinetics_fitting.plotting import plot_all, plot_k_vs_concentration
    from kinetics_fitting.utils import build_raw_data_lookup

    from pathlib import Path
    import os
    import pandas as pd
    import numpy as np

    pkg_dir = Path(__file__).resolve().parent
    eg_parent = pkg_dir.parent
    eg_parent = eg_parent.parent
    examples_folder = eg_parent / "examples"
    result_folder = examples_folder / "Python Plots Rate Constant"
    result_folder.mkdir(exist_ok=True)

    # --- MANUALLY DEFINE: labels
    salt_conc = ['Time (s)', 'Intensity (a.u.)'] #column_names
    salt_conc = [str(i) for i in salt_conc]

    rep = ['_540', '_650'] #excited wavelength ; emitted wavelength
    FS_label = [f'{salt}{inv}' for inv in rep for salt in salt_conc]

    print(FS_label)
    print(len(FS_label))

    # -----------------------------------
    ## Step 1: Make the labels for each column of the data
    name_of_data = 'KINETICS'
    number_of_replicates = 3
    importfiles = [examples_folder / f'{name_of_data}-{i:02d}.csv' for i in range(0, number_of_replicates)]

    data_dict = {}

    for file in importfiles:
        data = pd.read_csv(file)
        # set headers as the first row
        data.columns = data.iloc[0]
        # change the column names to the FS_label
        data.columns = FS_label
        # drop first row
        data = data.drop(0)
        #convert all columns to numeric
        data = data.apply(pd.to_numeric)
        # Store in dictionary using a clean key name
        key_name = file.stem  # Use stem to get the filename without extension from Path object
        data_dict[key_name] = data

    # Assign variables for easy access. This will depend on the number of files in a folder for any experiment.
    r5um_01, r5um_02, r8um_01 = data_dict.values()

    # Display first two rows of rep1
    print(r5um_01.head(5))

    r5um_01 = wavelengthMerger(r5um_01)
    r5um_02 = wavelengthMerger(r5um_02)
    r8um_01 = wavelengthMerger(r8um_01)
    # ... etc.

    # ------ plot graphs
    data_dict = {'r5um_01': r5um_01, 'r5um_02': r5um_02, 'r8um_01': r8um_01}
    names = list(data_dict)
    plot_all(data_dict, names, result_folder)

    # ------
    r5um_01 = waveExtractor(r5um_01)
    r5um_02 = waveExtractor(r5um_02)
    r8um_01 = waveExtractor(r8um_01)

    # ----- MANUALLY DEFINE regions of interest
    r501 = r5um_01[(r5um_01['Time'] >= 80) & (r5um_01['Time'] < 341.6620085)] #file 09
    r502 = r5um_02[(r5um_02['Time'] >= 65.56375122) & (r5um_02['Time'] < 367.1187439)] #file 10 #file 12 ; last time is basically till the end of the file.
    r801 = r8um_01[(r8um_01['Time'] >= 64.56200244) & (r8um_01['Time'] < 242.1649933)] #file 13

    td501 = r501['Time'].max() - r501['Time'].min()
    td502 = r502['Time'].max() - r502['Time'].min()
    td801 = r801['Time'].max() - r801['Time'].min()
    tdiff_list = [td501, td502, td801]

    # A0 is the first intensity value in the regime i.e.
    r501_A0 = r501['Intensity'].iloc[0]
    r501_650 = r501['Intensity'].iloc[1]

    r502_A0 = r502['Intensity'].iloc[0]
    r502_650 = r502['Intensity'].iloc[1]

    r801_A0 = r801['Intensity'].iloc[0]
    r801_650 = r801['Intensity'].iloc[1]

    A0_list = [r501_A0, r502_A0]

    # ----- fit reaction segments
    r101_fit = fit_reaction_segments(r501, 540, r501_A0, '101_rep1', result_folder)
    r102_fit = fit_reaction_segments(r502, 540, r502_A0, '102_rep1', result_folder)
    r101_650_fit = fit_reaction_segments(r501, 650, r501_650, '101_rep1', result_folder)
    r102_650_fit = fit_reaction_segments(r502, 650, r502_650, '102_rep1', result_folder)
    r801_fit = fit_reaction_segments(r801, 540, r801_A0, '801_rep1', result_folder)
    r801_650_fit = fit_reaction_segments(r801, 650, r801_650, '801_rep1', result_folder)

    # ----
    data_folder = examples_folder
    data_list = []

    # Loop through all matching Excel files
    for file in data_folder.glob(f"{name_of_data}-*.xlsx"):
        df = pd.read_excel(file)
        df = df.drop(df.index[0])    # Drop first row
        df = df.reset_index(drop=True)
        df["source_file"] = file.stem
        data_list.append(df)

    # Combine all into one big DataFrame
    combined_df = pd.concat(data_list, ignore_index=True)

    #update sample1_ex450_em540 to time_540
    combined_df.rename(columns={'Sample1_Ex450_Em540': 'Time (s)_540', 'Sample1_Ex450_Em620': 'Time (s)_650'}, inplace=True)
    #rename also Unnamed :1 to Intensity (a.u.)_540 and Intensity (a.u.)_650
    combined_df.rename(columns={'Unnamed: 1': 'Intensity (a.u.)_540', 'Unnamed: 3': 'Intensity (a.u.)_650'}, inplace=True)

    # for each file where 01 is the last number in the excel name add '1' to a column called 'concentration' ; if 02 add 0.5 ; if 03 add 0.5 ;...
    #r1um_01, r500nm_stopped, r500nm_01, r500nm_02, r2um_01, r2um_02, r5um_01, r5um_02, r5um_03, r2um_03, r1um_02, r8um_01, r8um_02, r8um_03, r500nm_03 = data_dict.values()
    # Extract the file number (e.g. '01', '02', ...) and convert to integer
    combined_df["file_num"] = combined_df["source_file"].str.extract(r"(\d+)$").astype(int)

    # Map the file number to the corresponding concentration
    conc_map = {1: 5.0, 2: 5.0}
    # and do the same for replicate
    rep_map = {1: 1, 2: 2}

    combined_df["Concentration (uM)"] = combined_df["file_num"].map(conc_map)
    combined_df["Replicate"] = combined_df["file_num"].map(rep_map)
    combined_df = pd.concat(data_list, ignore_index=True)
    combined_df.rename(columns={'Sample1_Ex450_Em540': 'Time (s)_540', 'Sample1_Ex450_Em620': 'Time (s)_650', 'Unnamed: 1': 'Intensity (a.u.)_540',
      'Unnamed: 3':'Intensity (a.u.)_650',}, inplace=True)
    print(combined_df.head())
    lookup = build_raw_data_lookup(combined_df)

    dfrates = dataExtractor(result_folder, lookup)

    # ----- convert dfrates to the same order as A0_list ; names 2ndfit_540_802_rep1
    dfrates['A0'] = np.nan
    dfrates.loc['540_r101_rep1', 'A0'] = r501_A0
    dfrates.loc['650_r101_rep1', 'A0'] = r501_650
    dfrates.loc['540_r102_rep1', 'A0'] = r502_A0
    dfrates.loc['650_r102_rep1', 'A0'] = r502_650
    dfrates.loc['540_r801_rep1', 'A0'] = r801_A0
    dfrates.loc['650_r801_rep1', 'A0'] = r801_650

    dfrates['tdiff (s)'] = np.nan
    dfrates.loc['540_r101_rep1', 'tdiff (s)'] = td501
    dfrates.loc['650_r101_rep1', 'tdiff (s)'] = td501
    dfrates.loc['540_r102_rep1', 'tdiff (s)'] = td502
    dfrates.loc['650_r102_rep1', 'tdiff (s)'] = td502
    dfrates.loc['540_r801_rep1', 'tdiff (s)'] = td801
    dfrates.loc['650_r801_rep1', 'tdiff (s)'] = td801

    dfrates = dfrates.apply(pd.to_numeric, errors='ignore')


    # ---- compute inflection time and max derivative
    numeric_cols = [
        'A1', 'A2', 'c', 'p', 'S', 
        'A1_cov', 'A2_cov', 'c_cov', 'p_cov', 'S_cov']

    dfrates[numeric_cols] = dfrates[numeric_cols].apply(pd.to_numeric, errors="coerce")

    dfrates[['inflection_time (s)', 'max_deriv (M/s)']] = dfrates.apply(
        lambda row: pd.Series(compute_inflection(row['A1'], row['A2'], row['c'], row['p'], row['S'])),
        axis=1
    )

    dfrates['inflection_time (s)'] = dfrates['inflection_time (s)'].apply(lambda x: f"{x:.2f}" if not pd.isnull(x) else "0")
    dfrates['max_deriv (M/s)'] = dfrates['max_deriv (M/s)'].apply(lambda x: f"{x:.2e}" if not pd.isnull(x) else "0") # max derivative IS Th ; set 0 as NaN so we dont have to worry about strings

    # ensure inflection time and max deriv are numeric
    dfrates['inflection_time (s)'] = pd.to_numeric(dfrates['inflection_time (s)'], errors='coerce')
    dfrates['max_deriv (M/s)'] = pd.to_numeric(dfrates['max_deriv (M/s)'], errors='coerce')

    # turn k column into significant figures
    dfrates['k'] = dfrates['A1'] #we already fixed this earlier.
    dfrates['k'] = dfrates['k'].apply(lambda x: f'{x:.2e}')
    dfrates['k'] = pd.to_numeric(dfrates['k'], errors='coerce')

    # update C column to 'conc (uM)' and add the following entries
    dfrates['Concentration'] = ['5.0', '5.0', '5.0', '5.0', '8.0', '8.0'] # uM concentration
    dfrates['Concentration'] = pd.to_numeric(dfrates['Concentration'], errors='coerce')

    # add also column for wavelength; 13 x 540 followed by 13 x 650
    dfrates['Wavelength'] = ['540'] * number_of_replicates + ['650'] * number_of_replicates
    dfrates['Wavelength'] = pd.to_numeric(dfrates['Wavelength'], errors='coerce')

    # ----- intermediate save
    dfrates.to_csv(result_folder / 'reaction_rates.csv', index=True)

    # finally take the mean of the k values for each group, grouping by 'group' AND 'wavelength', and add it to the dataframe
    dfrates['mean_k'] = dfrates.groupby(['Concentration', 'Wavelength'])['k'].transform('mean')
    dfrates['mean_k'] = pd.to_numeric(dfrates['mean_k'], errors='coerce')
    # and the std of the k values for each group
    dfrates['std_k'] = dfrates.groupby(['Concentration', 'Wavelength'])['k'].transform('std')
    dfrates['std_k'] = pd.to_numeric(dfrates['std_k'], errors='coerce')

    # take also the mean of max_deriv
    dfrates['mean_max_deriv'] = dfrates.groupby(['Concentration', 'Wavelength'])['max_deriv (M/s)'].transform('mean')
    dfrates['mean_max_deriv'] = pd.to_numeric(dfrates['mean_max_deriv'], errors='coerce')
    # take also the std of max_deriv
    dfrates['std_max_deriv'] = dfrates.groupby(['Concentration', 'Wavelength'])['max_deriv (M/s)'].transform('std')
    dfrates['std_max_deriv'] = pd.to_numeric(dfrates['std_max_deriv'], errors='coerce')

    # ---- final save
    dfrates.to_csv(result_folder / f'reaction_rates-mean.csv', index=True)

    # --- plotting whether rate constants are actually CONSTANT
    plot_k_vs_concentration(dfrates, wavelength=540, result_folder=result_folder)
    plot_k_vs_concentration(dfrates, wavelength=650, result_folder=result_folder)


if __name__ == "__main__":
    main()