# import packages
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from matplotlib.ticker import ScalarFormatter #this allows setting scientific notation

from kinetics_fitting.utils import annotate_ax

# pretty colors
mako = sns.color_palette('viridis', n_colors=6)
mako = mako[::-1] #reverse the palette
#sns.palplot(mako)
#plt.show()

# plot line graph for each replicate, using 'Group' as hue to separate the two measurements
def plotLineGraph(rep_df, title, filename, result_folder):
    fig, ax = plt.subplots()
    sns.lineplot(data=rep_df, x='Time', y='Intensity', hue='Group', palette=mako)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Intensity (a.u.)')

    ax.set_ylim(0, 200) #set limit to 200
    ax.legend(labels=['ex: 450 / em: 540', 'ex: 450 / em: 650'])
    annotate_ax(ax, title)
    # save plot as png
    plt.savefig(result_folder / f'{filename}.png', dpi=300, bbox_inches='tight')


def plot_all(data_dict, names, result_folder):
    for name in names:
        df = data_dict.get(name)
        if df is None:
            continue
        conc, rep = name.split('_')
        rep = rep  # e.g. "02"
        out_name = f"{conc[1:]}-rep-{rep}"
        plotLineGraph(df,
                      "Titration of ssDNA into SPN conjugated particles",
                      out_name,
                      result_folder)

# plot for each also 1/[A] concentration vs k and add a linear regression line
def plot_k_vs_concentration(dfrates, wavelength, result_folder):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    dfrates['Concentration'] = pd.to_numeric(dfrates['Concentration'])
    # do 1/concentration for the x-axis
    dfrates['Concentration'] = 1 / dfrates['Concentration']

    # and filter dfrates so that only values for 540 nm and 650 nm are shown
    dfrates = dfrates[dfrates['Wavelength'] == wavelength]

    # Plotting
    sns.scatterplot(data=dfrates, x='Concentration', y='k', alpha=0.5, ax=ax, legend=False)
    
    # Fit a linear regression line
    model = LinearRegression()
    X = dfrates[['Concentration']]
    y = dfrates['k']
    model.fit(X, y)
    
    # Predict values for the line
    x_fit = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_fit = model.predict(x_fit)
    
    # Plot the regression line
    ax.plot(x_fit, y_fit, color='red')
    # add equation and r^2 value to legend
    ax.legend(title=f'k = {model.coef_[0]:.2e} * [A] + {model.intercept_:.2e} \nR² = {model.score(X, y):.2f}')
    
    # Set labels and title
    ax.set_xlabel('1/[A] (µM⁻¹)')
    ax.set_ylabel('k (s⁻¹)')
    ax.set_title(f'k vs 1/[A] for Inc25-AF594 hybridizing to F8BT-Sub35 for {wavelength} nm')
    
    ax.set_ylim(1e2, 5e7)
    #ax.set_yscale('log')  # Set y-axis to logarithmic scale and show scientific notation
    # Set y-axis to scientific notation
    
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #ax.yaxis.get_major_formatter().set_powerlimits(( -4, 0 ))
    
    # Save figure as png
    plt.savefig(result_folder / f'k_vs_concentration_{wavelength}.png', dpi=300, bbox_inches='tight')
    plt.show()