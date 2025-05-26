# import packages
from pathlib import Path
import pandas as pd 
import re #regex package can be used to parse long strings
from .fitting import alt_five_param_logistic_equation

# merging time (s)_540 and time (s)_650 and intensity columns into one, + creating a new column for 'group' with either 540 or 650 as measurement
def wavelengthMerger(rep_df):
    df_540 = pd.DataFrame({
        'Time': rep_df['Time (s)_540'],
        'Intensity': rep_df['Intensity (a.u.)_540'],
        'Group': 'ex:450/em:540'})

    df_650 = pd.DataFrame({
        'Time': rep_df['Time (s)_650'],
        'Intensity': rep_df['Intensity (a.u.)_650'],
        'Group': 'ex:450/em:650'})

    result = pd.concat([df_540, df_650], ignore_index=True)
    # convert time to numeric
    result['Time'] = pd.to_numeric(result['Time'])
    #sort by time
    result = result.sort_values(by='Time')

    return result


# add a second column called Wavelength to each dataframe, that extracts the last three numbers from Group
def waveExtractor(rep_df):
    rep_df['Wavelength'] = rep_df['Group'].str.extract(r'(\d+$)')
    #and ensure it is numeric
    rep_df['Wavelength'] = pd.to_numeric(rep_df['Wavelength'])
    return rep_df


def dataExtractor(result_folder: Path, raw_data_lookup: dict) -> pd.DataFrame:
    results = {}
    
    # file structure: 2ndfit_650_502_rep1
    for file in result_folder.glob("2ndfit_*.txt"):
        #print("Found file:", file)
        with open(file, "r") as f:
            content = f.read()  # read entire text file

        try:
            filename = file.stem.replace("2ndfit_", "")
            params_match = re.search(r'params:\s*\[([^]]+)\]', content, flags=re.DOTALL)
            if not params_match:
                raise ValueError("Could not find 'params' array in text.")

            params_str = params_match.group(1)
            params_str = params_str.replace("[", "").replace("]", "").replace(",", "")
            tokens = params_str.split() #no argument splits on all whitespaces
            #print("DEBUC:G: tokens:", tokens)
            params = [float(x) for x in tokens]
            if len(params) != 5:
                raise ValueError(f"Expected 5 parameters, got {len(params)}: {params}")

            A1, A2, c, p_val, S = params
            # A1 now contains the rate constant.

            cov_match = re.search(r'covariance:\s*(\[\[.*\]\])', content, flags=re.DOTALL)
            if not cov_match:
                raise ValueError("Could not find 'covariance' array in text.")

            cov_str = cov_match.group(1)
            #print("DEBUC:G: cov_str:", cov_str)
            cov_str = cov_str.replace("[", '').replace("]", "").replace(",", "")
            cov_tokens = cov_str.split()
            cov_nums = [float(x) for x in cov_tokens]

            if len(cov_nums) != 25:
                raise ValueError(f"Expected at least 25 covariance values, got {len(cov_nums)}")
            
            # Suppose the diagonal is the first 5 entries (depending on shape):
            A1_cov = cov_nums[0]
            A2_cov = cov_nums[6]
            c_cov = cov_nums[12]
            p_cov = cov_nums[18]
            S_cov = cov_nums[24]

            # -- Add RSSE Calculation --
            if filename in raw_data_lookup:
                time_data = raw_data_lookup[filename]["Time (s)_540"]
                intensity_data = raw_data_lookup[filename]["Intensity (a.u.)_540"]

            # Normalize time as done in fit (shift so it starts at zero)
                time_norm = time_data - time_data.min()
                y_pred = alt_five_param_logistic_equation(time_norm, A1, A2, c, p_val, S)

            # store in dictionary
            results[filename] = [A1, A2, c, p_val, S, A1_cov, A2_cov, c_cov, p_cov, S_cov]

        except Exception as e:
            print(f"Error processing {file}: {e}")

    df = pd.DataFrame.from_dict(
        results, 
        orient='index', 
        columns=["A1", "A2", "c", "p", "S", "A1_cov", "A2_cov", "c_cov", "p_cov", "S_cov"]
    )

    return df