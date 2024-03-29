import seaborn as sns
import matplotlib.pyplot as plt
#import plotly.express as px
#import plotly.graph_objects as go
#import plotly.io as pio
import datetime
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import scipy
from scipy import stats
import numpy as np
import scanpy as sc
from matplotlib.lines import Line2D
import anndata
from matplotlib import rcParams
import pandas as pd


def make_plot_custom(embedding, colouring, save_dir=False, file_name="file_name", name="Emb type", description="details"):
    # Set the background to white
    sns.set(style="whitegrid", rc={"figure.figsize": (18, 12),'figure.dpi': 300, "axes.facecolor": "white", "grid.color": "white"})
    
    # Create a custom palette for the treatments of interest
    unique_treatments = set(embedding[colouring])
    custom_palette = sns.color_palette("hls", len(unique_treatments))
    color_dict = {treatment: color for treatment, color in zip(unique_treatments, custom_palette)}
    
    # Make the "Control" group grey
    if "DMSO_0.1%" in color_dict:
        color_dict["DMSO_0.1%"] = "lightgrey"
    
    # Create a size mapping
    size_dict = {treatment: 20 if treatment != "DMSO_0.1%" else 8 for treatment in unique_treatments}
    embedding['size'] = embedding[colouring].map(size_dict)
    
    # Create the scatter plot
    sns_plot = sns.scatterplot(data=embedding, x="UMAP 1", y="UMAP 2", hue=colouring, size='size', palette=color_dict, sizes=(8, 25), linewidth=0.1, alpha=0.9)
    
    plt.suptitle(f"{name}_{file_name}", fontsize=16)
    sns_plot.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    sns_plot.set_title("CLS Token embedding of "+str(len(embedding))+" cells" + " \n"+description, fontsize=12)
    sns.move_legend(sns_plot, "lower left", title='Treatments', prop={'size': 10}, title_fontsize=12, markerscale=0.5)
    
    # Remove grid lines
    sns.despine(bottom=True, left=True)
    
    if save_dir == True:
        # Save the figure with the specified DPI
        sns_plot.figure.savefig(f"{save_dir}{file_name}{name}.png", dpi=600)  # Changed DPI to 600
        sns_plot.figure.savefig(f"{save_dir}pdf_format/{file_name}{name}.pdf", dpi=600)  # Changed DPI to 600
    
    plt.show()



def make_plot_custom_plotly(embedding, colouring, save_dir=False, file_name="file_name", name="Emb type", description="details"):
    if isinstance(embedding, pl.DataFrame):
        embedding = embedding.to_pandas()
    # Create a custom palette for the treatments of interest
    unique_treatments = set(embedding[colouring])
    custom_palette = px.colors.qualitative.Set1[:len(unique_treatments)]
    color_dict = {treatment: color for treatment, color in zip(unique_treatments, custom_palette)}
    
    # Make the "Control" group grey
    if "DMSO_0.1%" in color_dict:
        color_dict["DMSO_0.1%"] = "lightgrey"
    
    # Create a size mapping
    size_dict = {treatment: 6 if treatment != "DMSO_0.1%" else 3 for treatment in unique_treatments}
    embedding['size'] = embedding[colouring].map(size_dict)
    
    # Create a Plotly scatter plot
    fig = px.scatter(embedding, x="UMAP1", y="UMAP2", color=colouring, size='size', color_discrete_map=color_dict, size_max=6, opacity=0.9, width=1000, height=700)

    # Customize the plot
    fig.update_layout(
        title={
            'text': f"{name} {file_name} - UMAP embedding of {len(embedding)} points \n{description}",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        plot_bgcolor='white',
        showlegend=True,
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        xaxis_showgrid=False,
        yaxis_showgrid=False
    )
    
    # Remove axis ticks and labels
    fig.update_xaxes(showticklabels=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, zeroline=False)

    # Save the figure if required
    if save_dir:
        fig.write_image(f"{save_dir}{file_name}{name}.png")
        fig.write_image(f"{save_dir}pdf_format/{file_name}{name}.pdf")
    
    # Show the plot
    fig.show()




def make_plot_custom_plotly_print(embedding, colouring, save_dir=False, file_name="file_name", name="Emb type", description="details"):
    
    # Define your custom color mapping based on conditions
    def get_color(val):
        if "_1" in val:
            return "#28B6D2"
        elif "_5" in val:
            return "#e96565"
        elif val == "DMSO_0.1%":
            return "lightgrey"
        else:
            return "grey"  # A default color for treatments that don't fit any of the conditions
    
    unique_treatments = set(embedding[colouring])
    color_dict = {treatment: get_color(treatment) for treatment in unique_treatments}
    low_conc = embedding[embedding[colouring].str.contains("_1")][colouring].unique()[0]
    high_conc = embedding[embedding[colouring].str.contains("_5")][colouring].unique()[0]
    # Create a size mapping
    size_dict = {treatment: 8 if treatment != "DMSO_0.1%" else 5 for treatment in unique_treatments}
    embedding['size'] = embedding[colouring].map(size_dict)
    
    fig = go.Figure()
    df_dmso = embedding[embedding[colouring] == "DMSO_0.1%"]
    fig.add_trace(
        go.Scatter(
            x=df_dmso["UMAP 1"],
            y=df_dmso["UMAP 2"],
            mode="markers",
            marker=dict(
                color=color_dict["DMSO_0.1%"],
                size=df_dmso['size']
            ),
            name="DMSO_0.1%"
        )
    )

    # Add "_1" dots
    df_1 = embedding[embedding[colouring].str.contains("_1")]
    fig.add_trace(
        go.Scatter(
            x=df_1["UMAP 1"],
            y=df_1["UMAP 2"],
            mode="markers",
            marker=dict(
                color=color_dict[low_conc],
                size=df_1['size']
            ),
            opacity=1,
            name=str(low_conc))
        )
    

    # Add "_5" dots
    df_5 = embedding[embedding[colouring].str.contains("_5")]
    fig.add_trace(
        go.Scatter(
            x=df_5["UMAP 1"],
            y=df_5["UMAP 2"],
            mode="markers",
            marker=dict(
                color=color_dict[high_conc],
                size=df_5['size']
            ),
            opacity=1,
            name=str(high_conc)
        )
    )

    # Customize the plot
    fig.update_layout(
        width = 1200,
        height = 900,
        title={
            'text': f"{name}_{file_name} embedding of {len(embedding)} cells \n{description}",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        plot_bgcolor='white',
        showlegend=True,
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        xaxis_showgrid=False,
        yaxis_showgrid=False
    )
    
    # Remove axis ticks and labels
    fig.update_xaxes(showticklabels=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, zeroline=False)

    # Save the figure if required
    if save_dir:
        #pio.write_image(fig, f"{save_dir}{file_name}{name}.png", scale=3)
        pio.write_image(fig, f"{file_name}{name}.pdf", scale=3)
    
    # Show the plot
    else:
        fig.show()



def make_jointplot(embedding, dmso_name, colouring, cmpd, save_path=None):
    
    # Generate a color palette based on unique values in the colouring column
    unique_treatments = embedding[colouring].unique()
    palette = sns.color_palette("Set2", len(unique_treatments))
    color_map = dict(zip(unique_treatments, palette))
    
    # Adjust colors and transparency if colouring is 'Metadat_cmpdName'
    if colouring == 'Metadata_cmpdName':
        if dmso_name in color_map:
            color_map[dmso_name] = 'lightgrey'
    
    embedding['color'] = embedding[colouring].map(color_map)
    point_size = 5
    embedding['size'] = point_size
    
    # Increase the DPI for displaying
    plt.rcParams['figure.dpi'] = 300
    
    # Create the base joint plot
    g = sns.JointGrid(x='UMAP1', y='UMAP2', data=embedding, height=10)

    # Plot KDE plots for each category
    for treatment in unique_treatments:
        subset = embedding[embedding[colouring] == treatment]
        
        sns.kdeplot(x=subset["UMAP1"], ax=g.ax_marg_x, fill=True, color=color_map[treatment], legend=False)
        sns.kdeplot(y=subset["UMAP2"], ax=g.ax_marg_y, fill=True, color=color_map[treatment], legend=False)

    # Plot the scatter plots
    for treatment in unique_treatments:
        subset = embedding[embedding[colouring] == treatment]
        alpha_val = 0.3 if treatment == dmso_name and colouring == 'Metadat_cmpdName' else 0.5
        g.ax_joint.scatter(subset["UMAP1"], subset["UMAP2"], c=subset['color'], s=subset['size'], label=treatment, alpha=alpha_val, edgecolor='white', linewidth=0.5)
    
    g.ax_joint.set_title(cmpd)
    legend = g.ax_joint.legend(fontsize=10)
    legend.get_frame().set_facecolor('white')
    # Display the plot
    

    
    if save_path != None:
        current_time = datetime.datetime.now()
        timestamp = current_time.strftime("%Y%m%d_%H%M%S")
        g.savefig(f"{save_path}.png", dpi=300)

    plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

def make_jointplot_anndata(adata, dmso_name, colouring, cmpd, save_path=None):
    # Extract UMAP data from .obsm
    umap_data = pd.DataFrame(adata.obsm['X_umap'], columns=['UMAP1', 'UMAP2']).reset_index()

    # Join with metadata from .obs
    embedding = pd.concat([umap_data,pd.DataFrame(adata.obs).reset_index()], axis = 1)
    embedding = embedding.reset_index()
    embedding["Metadata_cmpdNameConc2"] = embedding["Metadata_cmpdName"].astype(str) + "_" + embedding["Metadata_cmpdConc"].astype(str)
    # Generate a color palette based on unique values in the colouring column
    unique_treatments = embedding[colouring].unique()
    palette = sns.color_palette("Set2", len(unique_treatments))
    color_map = dict(zip(unique_treatments, palette))
    
    # Adjust colors and transparency if colouring is 'Metadata_cmpdName'
    if colouring == colouring:
        if dmso_name in color_map:
            color_map[dmso_name] = 'lightgrey'
    
    embedding['color'] = embedding[colouring].map(color_map)
    point_size = 3
    embedding['size'] = point_size
    
    # Increase the DPI for displaying
    plt.rcParams['figure.dpi'] = 300
    
    # Create the base joint plot
    g = sns.JointGrid(x='UMAP1', y='UMAP2', data=embedding, height=10)

    # Plot KDE plots for each category
    for treatment in unique_treatments:
        subset = embedding[embedding[colouring] == treatment]
        
        sns.kdeplot(x=subset["UMAP1"], ax=g.ax_marg_x, fill=True, color=color_map[treatment], legend=False)
        sns.kdeplot(y=subset["UMAP2"], ax=g.ax_marg_y, fill=True, color=color_map[treatment], legend=False)

    # Plot the scatter plots
    for treatment in unique_treatments:
        subset = embedding[embedding[colouring] == treatment]
        alpha_val = 0.3 if treatment == dmso_name and colouring == 'Metadata_cmpdName' else 0.5
        g.ax_joint.scatter(subset["UMAP1"], subset["UMAP2"], c=subset['color'], s=subset['size'], label=treatment, alpha=alpha_val, edgecolor='white', linewidth=0.5)
    
    g.ax_joint.set_title(cmpd)
    #legend = g.ax_joint.legend(fontsize=10)
    #legend.get_frame().set_facecolor('white')
    legend_elements = [Line2D([0], [0], marker='o', linestyle= "None", color=color_map[treatment], label=treatment, markersize=5, markerfacecolor=color_map[treatment], alpha=1) for treatment in unique_treatments]
    legend = g.ax_joint.legend(handles=legend_elements, fontsize=10, title=colouring)
    legend.get_frame().set_facecolor('white')
    
    if save_path != None:
        current_time = datetime.datetime.now()
        timestamp = current_time.strftime("%Y%m%d_%H%M%S")
        g.savefig(f"{save_path}_{timestamp}.png", dpi=300)

    plt.show()


def join_plot_dmso(embedding, plate_column, save_path=None):
    
    # Define the colors
    DMSO_color = 'black'
    other_color = 'lightgrey'

    # Generate a color palette based on unique values in the plate column for DMSO
    unique_plates = embedding[embedding['Metadata_cmpdName'] == 'DMSO'][plate_column].unique()
    palette = sns.color_palette("husl", len(unique_plates))
    plate_color_map = dict(zip(unique_plates, palette))
    
    # Map colors based on condition
    embedding['color'] = embedding.apply(lambda x: plate_color_map[x[plate_column]] if x['Metadata_cmpdName'] == 'DMSO' else other_color, axis=1)
    
    # Define the size for each point
    point_size = 20
    embedding['size'] = embedding.apply(lambda x: 50 if x['Metadata_cmpdName'] == 'DMSO' else point_size, axis=1)
    
    # Increase the DPI for displaying
    plt.rcParams['figure.dpi'] = 300
    
    # Create the base joint plot
    g = sns.JointGrid(x='UMAP1', y='UMAP2', data=embedding, height=10)
    
    subset_other = embedding[embedding['Metadata_cmpdName'] != 'DMSO']
    g.ax_joint.scatter(subset_other["UMAP1"], subset_other["UMAP2"], c=other_color, s=subset_other['size'], label="Others", alpha=0.5, edgecolor='white', linewidth=0.5)
    # Plot KDE plots for DMSO by plate
    for plate in unique_plates:
        subset = embedding[(embedding[plate_column] == plate) & (embedding['Metadata_cmpdName'] == 'DMSO')]
        sns.kdeplot(x=subset["UMAP1"], ax=g.ax_marg_x, fill=True, color=plate_color_map[plate], legend=False)
        sns.kdeplot(y=subset["UMAP2"], ax=g.ax_marg_y, fill=True, color=plate_color_map[plate], legend=False)

    # Plot the scatter plots
    for plate in unique_plates:
        subset = embedding[(embedding[plate_column] == plate) & (embedding['Metadata_cmpdName'] == 'DMSO')]
        g.ax_joint.scatter(subset["UMAP1"], subset["UMAP2"], c=subset['color'], s=subset['size'], label=plate, alpha=0.7, edgecolor='white', linewidth=0.5)
    
    # Plot grey points for other treatments
   

    g.ax_joint.set_title('DMSO by Plate')
    g.ax_joint.legend()

    # Display the plot
    plt.show()

    # Optionally, save the plot with high DPI
    if save_path:
        g.savefig(save_path, dpi=300)

def run_umap(df, feat):
    reducer = umap.UMAP(n_neighbors=n_neighbors_value, metric = "euclidean", min_dist=min_dist_value)
    embedding = reducer.fit_transform(df[feat])
    umap_df = pd.DataFrame(data=embedding, columns=["UMAP1", "UMAP2"])
    umap_df[['Plate', 'Well', 'Site']] = df[['Plate', 'Well', 'Site']].reset_index(drop = True)
    umap_df = pd.merge(umap_df, meta, on=["Plate", "Well", "Site"])
    return umap_df



def find_file_with_string(directory, string):
    """
    Finds a file in the specified directory that contains the given string in its name.

    Args:
    directory (str): The directory to search in.
    string (str): The string to look for in the file names.

    Returns:
    str: The path to the first file found that contains the string. None if no such file is found.
    """
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"The directory {directory} does not exist.")
        return None

    # Iterate through all files in the directory
    for file in os.listdir(directory):
        if string in file:
            return os.path.join(directory, file)

    # Return None if no file is found
    return None




def make_jointplot_seaborn_benchmark(embedding, colouring, cmpd, samp_meth, samp_rate, overlay=False, overlay_df=None):
    
    def get_color(val):
        if "[DMSO]" in val:
            return "lightgrey"
        else:
            return "#e96565"
    
    def get_size(val):
        return 20 if val != "[DMSO]" else 10
    
    embedding['color'] = embedding[colouring].apply(get_color)
    embedding['size'] = embedding[colouring].apply(get_size)

    all_treatments = list(embedding[colouring].unique())
    sorted_treatments = all_treatments.copy()
    specific_value = '[DMSO]'
    if specific_value in sorted_treatments:
        sorted_treatments.remove(specific_value)
    sorted_treatments.insert(0, specific_value)

    g = sns.JointGrid(x='UMAP1', y='UMAP2', data=embedding, height=10)

    for treatment in sorted_treatments:
        subset = embedding[embedding[colouring] == treatment]
        
        sns.kdeplot(x=subset["UMAP1"], ax=g.ax_marg_x, fill=True, color=get_color(treatment), legend=False)
        sns.kdeplot(y=subset["UMAP2"], ax=g.ax_marg_y, fill=True, color=get_color(treatment), legend=False)

    for treatment in sorted_treatments:
        subset = embedding[embedding[colouring] == treatment]
        g.ax_joint.scatter(subset["UMAP1"], subset["UMAP2"], c=subset['color'], s=subset['size'], label=f"{treatment} - {len(subset)} cells", alpha=0.7, edgecolor='white', linewidth=0.5)

    # Overlay additional points if the option is active
    if overlay and overlay_df is not None:
        overlay_df['color'] = overlay_df[colouring].apply(get_color)
        # Increase the size for the overlay points
        overlay_df['size'] = overlay_df[colouring].apply(lambda val: get_size(val) * 2)  
        
        for treatment in sorted_treatments:
            subset = overlay_df[overlay_df[colouring] == treatment]
            g.ax_joint.scatter(subset["UMAP1"], subset["UMAP2"], c=subset['color'], s=subset['size'], alpha=0.9, edgecolor='grey', linewidth=0.5)

    g.ax_joint.set_title(f"{cmpd} {samp_meth} rate: {samp_rate}")
    g.ax_joint.legend()
    filename = f"grit_umap_specs3k_{cmpd}_{samp_meth}_{samp_rate}.png"
    plt.savefig(os.path.join("Figures_SPECS3K", "sampling_benchmark",filename), dpi=300, bbox_inches='tight')
    #plt.show()


def make_jointplot_seaborn_RH30(embedding, colouring, cmpd):
    
    # Define your custom color mapping based on conditions
    def get_color(val):
        if "_1" in val:
            return "#28B6D2"
        elif "_5" in val:
            return "#e96565"
        elif val == "DMSO_0":
            return "lightgrey"
        else:
            return "grey"
    
    # Size mapping
    def get_size(val):
        return 20 if val != "DMSO_0" else 10
    
    embedding['color'] = embedding[colouring].apply(get_color)
    embedding['size'] = embedding[colouring].apply(get_size)
    
    # Create the base joint plot
    g = sns.JointGrid(x='UMAP1', y='UMAP2', data=embedding, height=10)

    # Plot KDE plots for each category
    for treatment in embedding[colouring].unique():
        subset = embedding[embedding[colouring] == treatment]
        
        sns.kdeplot(x=subset["UMAP1"], ax=g.ax_marg_x, fill=True, color=get_color(treatment), legend=False)
        sns.kdeplot(y=subset["UMAP2"], ax=g.ax_marg_y, fill=True, color=get_color(treatment), legend=False)

    # Ensure plotting order: first DMSO_0.1%, then _1, and finally _5
    for treatment in ["DMSO_0", "_1", "_5"]:
        subset = embedding[embedding[colouring].str.contains(treatment)]
        g.ax_joint.scatter(subset["UMAP1"], subset["UMAP2"], c=subset['color'], s=subset['size'], label=f"{treatment} - {len(embedding)} cells", alpha=0.7, edgecolor='white', linewidth=0.5)
    
    g.ax_joint.set_title(cmpd)
    g.ax_joint.legend()

    plt.show()


def make_jointplot_seaborn_specs(embedding, colouring, cmpd, overlay=False, overlay_df=None):
    
    def get_color(val):
        if "[DMSO]" in val:
            return "lightgrey"
        else:
            return "#e96565"
    
    def get_size(val):
        return 20 if val != "[DMSO]" else 10
    
    embedding['color'] = embedding[colouring].apply(get_color)
    embedding['size'] = embedding[colouring].apply(get_size)

    all_treatments = list(embedding[colouring].unique())
    sorted_treatments = all_treatments.copy()
    specific_value = '[DMSO]'
    if specific_value in sorted_treatments:
        sorted_treatments.remove(specific_value)
    sorted_treatments.insert(0, specific_value)

    g = sns.JointGrid(x='UMAP1', y='UMAP2', data=embedding, height=10)

    for treatment in sorted_treatments:
        subset = embedding[embedding[colouring] == treatment]
        
        sns.kdeplot(x=subset["UMAP1"], ax=g.ax_marg_x, fill=True, color=get_color(treatment), legend=False)
        sns.kdeplot(y=subset["UMAP2"], ax=g.ax_marg_y, fill=True, color=get_color(treatment), legend=False)

    for treatment in sorted_treatments:
        subset = embedding[embedding[colouring] == treatment]
        g.ax_joint.scatter(subset["UMAP1"], subset["UMAP2"], c=subset['color'], s=subset['size'], label=f"{treatment} - {len(subset)} cells", alpha=0.7, edgecolor='white', linewidth=0.5)

    # Overlay additional points if the option is active
    if overlay and overlay_df is not None:
        overlay_df['color'] = overlay_df[colouring].apply(get_color)
        # Increase the size for the overlay points
        overlay_df['size'] = overlay_df[colouring].apply(lambda val: get_size(val) * 2)  
        
        for treatment in sorted_treatments:
            subset = overlay_df[overlay_df[colouring] == treatment]
            g.ax_joint.scatter(subset["UMAP1"], subset["UMAP2"], c=subset['color'], s=subset['size'], alpha=0.9, edgecolor='grey', linewidth=0.5)

    g.ax_joint.set_title(cmpd)
    g.ax_joint.legend()
    plt.show()




def make_jointplot_seaborn_grit(embedding, colouring, cmpd, name, save = True, overlay=False, overlay_df=None, color_by_other_column=False, other_column=None):
    
    def get_continuous_color_map(data, cmap_name="viridis"):
        # Create a color map based on the range of the data
        cmap = plt.get_cmap(cmap_name)
        min_val, max_val = data.min(), data.max()
        norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
        color_map = cm.ScalarMappable(norm=norm, cmap=cmap)
        return color_map

    # Create a color map if coloring by another column
    if color_by_other_column and other_column is not None:
        color_map = get_continuous_color_map(embedding[other_column])
        embedding['color'] = embedding.apply(lambda row: color_map.to_rgba(row[other_column]) if row[colouring] != "[DMSO]" else "lightgrey", axis=1)
    else:
        color_map, color_dict = None, None
        embedding['color'] = embedding[colouring].apply(lambda val: "lightgrey" if "[DMSO]" in val else "#e96565")


    #embedding['color'] = embedding.apply(lambda row: "lightgrey" if "[DMSO]" in row[colouring] else color_map[row[other_column]], axis=1) if color_by_other_column and other_column is not None else embedding[colouring].apply(lambda val: "lightgrey" if "[DMSO]" in val else "#e96565")
    embedding['size'] = embedding[colouring].apply(lambda val: 20 if val != "[DMSO]" else 10)

    all_treatments = list(embedding[colouring].unique())
    sorted_treatments = all_treatments.copy()
    specific_value = '[DMSO]'
    if specific_value in sorted_treatments:
        sorted_treatments.remove(specific_value)
    sorted_treatments.insert(0, specific_value)

    g = sns.JointGrid(x='UMAP1', y='UMAP2', data=embedding, height=10)

    for treatment in sorted_treatments:
        subset = embedding[embedding[colouring] == treatment]
        color = subset['color'].iloc[0] if color_by_other_column and other_column is not None else "lightgrey" if "[DMSO]" in treatment else "#e96565"
        sns.kdeplot(x=subset["UMAP1"], ax=g.ax_marg_x, fill=True, color=color, legend=False)
        sns.kdeplot(y=subset["UMAP2"], ax=g.ax_marg_y, fill=True, color=color, legend=False)

    for treatment in sorted_treatments:
        subset = embedding[embedding[colouring] == treatment]
        g.ax_joint.scatter(subset["UMAP1"], subset["UMAP2"], c=subset['color'], s=subset['size'], label=f"{treatment} - {len(subset)} cells", alpha=0.7, edgecolor='white', linewidth=0.5)

    # Overlay additional points if the option is active
    if overlay and overlay_df is not None:
        overlay_df['color'] = overlay_df.apply(lambda row: "lightgrey" if "[DMSO]" in row[colouring] else color_map[row[other_column]], axis=1) if color_by_other_column and other_column is not None else overlay_df[colouring].apply(lambda val: "lightgrey" if "[DMSO]" in val else "#e96565")
        overlay_df['size'] = overlay_df[colouring].apply(lambda val: 20 if val != "[DMSO]" else 10)

        for treatment in sorted_treatments:
            subset = overlay_df[overlay_df[colouring] == treatment]
            g.ax_joint.scatter(subset["UMAP1"], subset["UMAP2"], c=subset['color'], s=subset['size'], alpha=0.9, edgecolor='grey', linewidth=0.5)

    g.ax_joint.set_title(cmpd)
    g.ax_joint.tick_params(axis='both', labelsize=10)
    g.ax_marg_x.tick_params(axis='x', labelsize=10)
    g.ax_marg_y.tick_params(axis='y', labelsize=10)
    legend = g.ax_joint.legend(fontsize=10)
    legend.get_frame().set_facecolor('white')
    #g.ax_joint.legend()

    if color_by_other_column and color_map is not None:
        cbar_ax = g.fig.add_axes([1, 0.25, 0.02, 0.5])
        plt.colorbar(color_map, cax=cbar_ax,orientation='vertical', label=other_column)
    filename = f"grit_umap_specs3k_{cmpd}_{name}.png"
    if save == True:
        plt.savefig(os.path.join("Figures_SPECS3K", filename), dpi=300, bbox_inches='tight')
    plt.show()



def make_jointplot_seaborn_density(embedding, colouring, cmpd, overlay=False, overlay_df=None):
    
    def get_color(val):
        if "[DMSO]" in val:
            return "lightgrey"
        else:
            return "#e96565"  # This color will be overridden for non-[DMSO] treatments
    
    def get_size(val):
        return 20 if val != "[DMSO]" else 10
    
    embedding['color'] = embedding[colouring].apply(get_color)
    embedding['size'] = embedding[colouring].apply(get_size)

    all_treatments = list(embedding[colouring].unique())
    sorted_treatments = all_treatments.copy()
    specific_value = '[DMSO]'
    if specific_value in sorted_treatments:
        sorted_treatments.remove(specific_value)
    sorted_treatments.insert(0, specific_value)

    g = sns.JointGrid(x='UMAP1', y='UMAP2', data=embedding, height=10)

    #cmap = plt.cm.viridis
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for treatment in sorted_treatments:
        subset = embedding[embedding[colouring] == treatment]
        
        if treatment != '[DMSO]':
            # Calculate density for non-[DMSO] treatments
            values = np.vstack([subset["UMAP1"], subset["UMAP2"]])
            kernel = stats.gaussian_kde(values)(values)
            colors = cmap(kernel)

            # Plot KDE for x and y axes
            sns.kdeplot(x=subset["UMAP1"], ax=g.ax_marg_x, fill=True, color=colors.mean(axis=0), legend=False)
            sns.kdeplot(y=subset["UMAP2"], ax=g.ax_marg_y, fill=True, color=colors.mean(axis=0), legend=False)

            # Scatter plot with density color
            g.ax_joint.scatter(subset["UMAP1"], subset["UMAP2"], c=kernel, s=subset['size'], cmap=cmap, label=f"{treatment} - {len(subset)} cells", alpha=0.7, edgecolor='white', linewidth=0.5)
        else:
            # Plot for [DMSO] treatment
            sns.kdeplot(x=subset["UMAP1"], ax=g.ax_marg_x, fill=True, color='lightgrey', legend=False)
            sns.kdeplot(y=subset["UMAP2"], ax=g.ax_marg_y, fill=True, color='lightgrey', legend=False)
            g.ax_joint.scatter(subset["UMAP1"], subset["UMAP2"], c='lightgrey', s=subset['size'], label=f"{treatment} - {len(subset)} cells", alpha=0.7, edgecolor='white', linewidth=0.5)
    # Overlay additional points if the option is active
    if overlay and overlay_df is not None:
        overlay_df['color'] = overlay_df[colouring].apply(get_color)
        overlay_df['size'] = overlay_df[colouring].apply(lambda val: get_size(val) * 2)  
        
        for treatment in sorted_treatments:
            subset = overlay_df[overlay_df[colouring] == treatment]
            g.ax_joint.scatter(subset["UMAP1"], subset["UMAP2"], c=subset['color'], s=subset['size'], alpha=0.9, edgecolor='grey', linewidth=0.5)

    #plt.colorbar(sm, ax=g.ax_joint, pad=0.05, aspect=10)

    fig = g.fig  # Get the figure of the JointGrid
    cbar_ax = fig.add_axes([0.93, 0.1, 0.02, 0.7])  # Add axes for the colorbar

    # Add colorbar to the figure, not the joint plot axes
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Relative density', rotation=270, labelpad=15) 

    # Adjust the figure to make space for the colorbar
    fig.subplots_adjust(right=0.9)

    g.ax_joint.set_title(cmpd)
    g.ax_joint.legend()
    plt.show()

def make_jointplot_pca(embedding, colouring, cmpd, save_path=None):
    
    # Generate a color palette based on unique values in the colouring column
    unique_treatments = embedding[colouring].unique()
    palette = sns.color_palette("Set2", len(unique_treatments))
    color_map = dict(zip(unique_treatments, palette))
    
    # Adjust colors and transparency if colouring is 'Metadat_cmpdName'
    if colouring == 'Metadata_cmpdName':
        if '[DMSO]' in color_map:
            color_map['[DMSO]'] = 'lightgrey'
    
    embedding['color'] = embedding[colouring].map(color_map)
    point_size = 10
    embedding['size'] = point_size
    
    # Increase the DPI for displaying
    plt.rcParams['figure.dpi'] = 300
    
    # Create the base joint plot
    g = sns.JointGrid(x='pc1', y='pc2', data=embedding, height=10)

    # Plot KDE plots for each category
    for treatment in unique_treatments:
        subset = embedding[embedding[colouring] == treatment]
        
        sns.kdeplot(x=subset["pc1"], ax=g.ax_marg_x, fill=True, color=color_map[treatment], legend=False)
        sns.kdeplot(y=subset["pc2"], ax=g.ax_marg_y, fill=True, color=color_map[treatment], legend=False)

    # Plot the scatter plots
    for treatment in unique_treatments:
        subset = embedding[embedding[colouring] == treatment]
        alpha_val = 0.3 if treatment == '[DMSO]' and colouring == 'Metadat_cmpdName' else 0.5
        g.ax_joint.scatter(subset["pc1"], subset["pc2"], c=subset['color'], s=subset['size'], label=treatment, alpha=alpha_val, edgecolor='white', linewidth=0.5)
    
    g.ax_joint.set_title(cmpd)
    legend = g.ax_joint.legend(fontsize=10)
    legend.get_frame().set_facecolor('white')

    # Display the plot
    

    
    if save_path != None:
        current_time = datetime.datetime.now()
        timestamp = current_time.strftime("%Y%m%d_%H%M%S")
        g.savefig(f"{save_path}.png", dpi=300)

    plt.show()


def make_plot_custom_pca(embedding, colouring, save_dir=False, file_name="", name="", description=""):
    # Set the background to white
    sns.set(style="whitegrid", rc={"figure.figsize": (18, 12),'figure.dpi': 300, "axes.facecolor": "white", "grid.color": "white"})
    
    # Create a custom palette for the treatments of interest
    unique_treatments = set(embedding[colouring])
    custom_palette = sns.color_palette("hls", len(unique_treatments))
    color_dict = {treatment: color for treatment, color in zip(unique_treatments, custom_palette)}
    
    # Make the "Control" group grey
    if "[DMSO]" in color_dict:
        color_dict["[DMSO]"] = "lightgrey"
    
    # Create a size mapping
    size_dict = {treatment: 20 if treatment != "[DMSO]" else 8 for treatment in unique_treatments}
    embedding['size'] = embedding[colouring].map(size_dict)
    
    # Create the scatter plot
    sns_plot = sns.scatterplot(data=embedding, x="pc1", y="pc2", hue=colouring, size='size', palette=color_dict, sizes=(8, 25), linewidth=0.1, alpha=0.9)
    
    plt.suptitle(f"{name}_{file_name}", fontsize=16)
    sns_plot.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    sns_plot.set_title("PCA of "+str(len(embedding))+" data points" + " \n"+description, fontsize=12)
    sns.move_legend(sns_plot, "lower left", title='Treatments', prop={'size': 10}, title_fontsize=12, markerscale=0.5)
    
    # Remove grid lines
    sns.despine(bottom=True, left=True)
    
    if save_dir == True:
        # Save the figure with the specified DPI
        sns_plot.figure.savefig(f"{save_dir}{file_name}{name}.png", dpi=600)  # Changed DPI to 600
        sns_plot.figure.savefig(f"{save_dir}pdf_format/{file_name}{name}.pdf", dpi=600)  # Changed DPI to 600
    
    plt.show()



def plot_treatment_scatter(df, x_coord_column, y_coord_column, treatment_column):
    """
    Create a grid of scatter plots of points based on their spatial coordinates, 
    with each subplot corresponding to a different treatment group.

    Parameters:
    df (DataFrame): The DataFrame containing the coordinate and treatment data.
    x_coord_column (str): The name of the column containing x coordinates.
    y_coord_column (str): The name of the column containing y coordinates.
    treatment_column (str): The name of the column containing treatment labels.
    """
    # Determine the number of treatment groups and setup the subplot grid
    treatments = df[treatment_column].unique()
    n_treatments = len(treatments)
    n_cols = 3  # number of columns
    n_rows = int(np.ceil(n_treatments / n_cols))  # calculate required number of rows

    # Create a figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False, dpi = 300)
    axes = axes.flatten()  # Flatten to 1D array for easy iteration

    # Loop through each treatment and create a subplot
    for i, treatment in enumerate(treatments):
        ax = axes[i]
        subset = df[df[treatment_column] == treatment]
        
        sns.scatterplot(data=subset, x=x_coord_column, y=y_coord_column, ax=ax,
                        hue="Metadata_cmpdName", palette="Set2", legend=False, s = 5)  # Remove individual legends
        
        ax.set_title(f'Treatment: {treatment}')
        ax.set_xlim(0, 2500)  # Adjust according to your data
        ax.set_ylim(0, 2500)  # Adjust according to your data

    # Hide any unused subplots
    for j in range(n_treatments, len(axes)):
        axes[j].axis('off')

    # Create a single legend outside the rightmost subplot
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.1, 1), ncol=1)
    
    plt.tight_layout()


    
"Scanpy Plotting functions"



def plot_umap_grid_colored(anndata_dict, color_by, n_cols=3):
    """
    Create a grid of UMAP plots from a dictionary of AnnData objects, colored by a specified column.
    
    Parameters:
    anndata_dict (dict): A dictionary of AnnData objects.
    color_by (str): Column name to color by.
    n_cols (int): Number of columns in the grid.
    """
    # Determine all unique categories across all AnnData objects
    anndata_dict = {k: v for k, v in anndata_dict.items() if k != 'all'}
    all_categories = set()
    for adata in anndata_dict.values():
        all_categories.update(adata.obs[color_by].astype(str))

    # Sort categories for consistent ordering and create color palette
    sorted_categories = sorted(list(all_categories))
    color_palette = sc.pl.palettes.default_20 # Use any large enough palette or define your own
    color_map = {cat: color_palette[i % len(color_palette)] for i, cat in enumerate(sorted_categories)}
    #color_map = {'big_dmso': '#1f77b4', 'small_dmso': '#ff7f0e', 'small_FLUP': '#279e68', 'big_FLUP': '#d62728', 'big_ETOP': '#aa40fc', 'small_ETOP': '#8c564b', 'big_TETR': '#e377c2', 'small_TETR': '#b5bd61', 'small_CA-O': '#17becf', 'big_CA-O': '#aec7e8', 'unassigned': '#ffbb78', 'BERB': '#98df8a', 'FEB': '#ff9896'}
    print(color_map)
    # Set up the figure for subplots
    n_rows = int(np.ceil(len(anndata_dict) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axs = axs.flatten()  # Flatten to make indexing axs easier

    all_handles = []
    all_labels = set()
    
    # Plot UMAP for each AnnData object
    for ax, (key, adata) in zip(axs, anndata_dict.items()):
        sc.pl.umap(adata, color=color_by, ax=ax, show=False, 
                   title=key, frameon=False,
                   palette=color_map,
                   legend_loc = "none")  # Apply the consistent color map

        handles, labels = ax.get_legend_handles_labels()
        all_handles.extend(handles)
        all_labels.update(labels)
        # Remove axis titles (optional, for cleaner look)
        ax.set_xlabel('')
        ax.set_ylabel('')

    # Hide any extra axes
    for i in range(len(anndata_dict), len(axs)):
        axs[i].axis('off')

    # Create an overall title
    plt.suptitle('UMAP Grid', fontsize=16)

    # Add a single legend outside the plots
    # Get handles and labels for legend from the last plot
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_single_umap_colored(adata, color_by):
    # Calculate the count of each category in the color_by column
    category_counts = adata.obs[color_by].value_counts()

    # Create a color palette
    #color_map = {'big_dmso': '#1f77b4', 'small_dmso': '#ff7f0e', 'small_FLUP': '#279e68', 'big_FLUP': '#d62728', 'big_ETOP': '#aa40fc', 'small_ETOP': '#8c564b', 'big_TETR': '#e377c2', 'small_TETR': '#b5bd61', 'small_CA-O': '#17becf', 'big_CA-O': '#aec7e8', 'unassigned': '#ffbb78', 'BERB': '#98df8a', 'FEB': '#ff9896'}
    color_map = {'[BERB]': '#1f77b4', '[CA-0]': '#ff7f0e', '[DMSO]': '#279e68', '[ETOP]': '#d62728', '[FENB]': '#aa40fc', '[FLUP]': '#8c564b', '[TETR]': '#e377c2'}
    # Create figure and axis for UMAP plot
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size as needed

    # Create UMAP plot
    sc.pl.umap(adata, color=color_by, ax=ax, show=False,
               title=f'UMAP colored by {color_by}', 
               frameon=False, legend_loc='none', 
               palette=color_map, s = 2)

    # Create a custom legend for all categories with counts
    legend_elements = [Line2D([0], [0], marker='o', color='w',
                              label=f"{cat} (n={category_counts[cat]})",
                              markerfacecolor=color_map[cat], markersize=10)
                       for cat in category_counts.index]

    # Place legend outside the plot to the right
    ax.legend(handles=legend_elements, title=color_by, loc='center left',
              bbox_to_anchor=(1, 0.5), ncol=1, fontsize='x-small')

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the rect parameter to make space for the legend
    plt.show()



def show_summary_stats(df):
    features = df.columns

# Plotting
    plt.figure(figsize=(12,6))

    # Mean line
    plt.plot(features, df.loc['mean'], label='Mean', color='blue')

    # 5th percentile line
    plt.plot(features, df.loc['5%'], label='5th Percentile', color='green')

    # 95th percentile line
    plt.plot(features, df.loc['95%'], label='95th Percentile', color='red')

    # Max values as dots
    plt.scatter(features, df.loc['max'], color='black', label='Max', s=5)  # s is the size of points
    plt.scatter(features, df.loc['min'], color='grey', label='Min', s=5)

    # Labels and title
    plt.xlabel('Features')
    plt.ylabel('Values')
    plt.title('Feature distributions')
    plt.xticks([])  # Rotate feature names for readability

    # Legend
    plt.legend()

    plt.tight_layout()  # Adjust layout
    plt.show()


def plot_grouped_feature_statistics(df, group_column, feature_columns):
    """
    Plot statistical summaries (mean, 5th, 95th percentiles, and max) of features for each group in the DataFrame.
    
    Parameters:
    df (DataFrame): The original pandas DataFrame with data.
    group_column (str): The name of the column to group by.
    feature_columns (list): List of columns to calculate statistics on.
    """
    # Grouping the DataFrame by the specified column
    grouped = df.groupby(group_column)

    # Determine the number of subplots needed
    n_groups = len(grouped)
    n_cols = 1  # You can adjust the number of columns per row
    n_rows = int(np.ceil(n_groups / n_cols))

    # Create a figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15 * n_cols, 10 * n_rows), squeeze=False)
    axes = axes.flatten()  # Flatten to 1D array for easy iteration

    for i, (group_name, group_data) in enumerate(grouped):
        # Calculating statistics for the group
        mean = group_data[feature_columns].mean()
        std = group_data[feature_columns].std()
        min_val = group_data[feature_columns].min()
        max_val = group_data[feature_columns].max()
        percentile_5 = group_data[feature_columns].quantile(0.05)
        percentile_95 = group_data[feature_columns].quantile(0.95)

        # Plotting on the ith subplot
        ax = axes[i]
        ax.plot(feature_columns, mean, label='Mean', color='blue')
        ax.plot(feature_columns, percentile_5, label='5th Percentile', color='green')
        ax.plot(feature_columns, percentile_95, label='95th Percentile', color='red')


        ax.set_title(f'Group: {group_name}')
        ax.set_xticks([])  # Remove x-axis labels

        if i == 0:  # Add legend to the first subplot as an example
            ax.legend()

    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def plot_clipped_heatmap(adata, max_val=10, min_val=-10, genes=None, groupby=None):
    """
    Plot a heatmap from clipped data of an AnnData object.

    Parameters:
    adata (AnnData): The original AnnData object.
    max_val (float): Maximum value to clip data to.
    min_val (float): Minimum value to clip data to.
    genes (list): List of gene names to be plotted. They should match the var_names in adata.
    groupby (str): Name of the observation annotation to group by (usually categorical).

    Returns:
    None: Displays a heatmap.
    """

    # Step 1: Make a copy of the AnnData object to avoid overwriting original data
    adata_copy = adata.copy()

    # Step 2: Clip the data in the X matrix of the copied AnnData object
    # Check if 'X' is dense or sparse and clip accordingly
    if isinstance(adata_copy.X, np.ndarray):
        adata_copy.X = np.clip(adata_copy.X, a_min=min_val, a_max=max_val)
    elif isinstance(adata_copy.X, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)):
        adata_copy.X.data = np.clip(adata_copy.X.data, a_min=min_val, a_max=max_val)
    else:
        raise TypeError("adata.X must be a numpy array or a scipy sparse matrix.")

    rcParams["figure.figsize"]  =(10,10)
    # Step 3: Use scanpy's pl.heatmap function to visualize the clipped data
    sc.pl.heatmap(adata_copy, var_names=genes, groupby=groupby, swap_axes= True, standard_scale = "obs")


def plot_treatment_scatter(df: pd.DataFrame(), x_coord_column, y_coord_column, treatment_column, xlim, ylim, x1, x2, y1, y2):
    """
    Create a grid of scatter plots of points based on their spatial coordinates, 
    with each subplot corresponding to a different treatment group.

    Parameters:
    df (DataFrame): The DataFrame containing the coordinate and treatment data.
    x_coord_column (str): The name of the column containing x coordinates.
    y_coord_column (str): The name of the column containing y coordinates.
    treatment_column (str): The name of the column containing treatment labels.
    """
    # Determine the number of treatment groups and setup the subplot grid
    treatments = df[treatment_column].unique()
    n_treatments = len(treatments)
    n_cols = 3  # number of columns
    n_rows = int(np.ceil(n_treatments / n_cols))  # calculate required number of rows

    # Create a figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False, dpi = 300)
    axes = axes.flatten()  # Flatten to 1D array for easy iteration

    # Loop through each treatment and create a subplot
    for i, treatment in enumerate(treatments):
        ax = axes[i]
        subset = df[df[treatment_column] == treatment]
        
        sns.scatterplot(data=subset, x=x_coord_column, y=y_coord_column, ax=ax,
                        hue="Metadata_cmpdName", palette="Set2", legend=False, s = 5)  # Remove individual legends
        

        ax.axvline(x=x1, color='red', linestyle='--')  # Vertical line at x1
        ax.axvline(x=x2, color='red', linestyle='--')  # Vertical line at x2
        ax.axhline(y=y1, color='red', linestyle='--')  # Horizontal line at y1
        ax.axhline(y=y2, color='red', linestyle='--')  # Horizontal line at y2
        ax.set_title(f'Treatment: {treatment}')

        ax.set_xlim(0, xlim)  # Adjust according to your data
        ax.set_ylim(0, ylim)  # Adjust according to your data

    # Hide any unused subplots
    for j in range(n_treatments, len(axes)):
        axes[j].axis('off')

    # Create a single legend outside the rightmost subplot
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.1, 1), ncol=1)
    
    plt.tight_layout()
    plt.show()
