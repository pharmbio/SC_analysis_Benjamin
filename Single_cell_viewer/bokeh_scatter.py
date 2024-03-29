from __future__ import annotations
from typing import *

import numpy as np
import pandas as pd
import uuid

from bokeh.models import *
from bokeh.plotting import *
from bokeh.layouts import *
from bokeh.palettes import *
from bokeh.transform import factor_cmap, linear_cmap
import bokeh.core.properties as properties

import viewer

from bokeh.models import Select, CustomJS, ColumnDataSource, Button, ColorBar, HoverTool, LinearColorMapper, CategoricalColorMapper, DataRange1d
from bokeh.io import output_notebook
from typing import Sequence, Literal
from bokeh.models import CheckboxGroup
from bokeh.models import CheckboxGroup, CustomJS
from bokeh.layouts import column

def bokeh_to_html(root, id: str):
    import bokeh.embed
    import bokeh.document
    import bokeh.themes
    import bokeh.embed
    import bokeh.document
    import bokeh.themes

    script, div = bokeh.embed.components(root, theme=bokeh.themes.built_in_themes['dark_minimal'])

    osd = viewer.Viewer([], height='100%').to_html()

    html = '''
        <style>
            .bokeh_scatter {
                box-sizing: border-box;
                display: grid;
                width: 100%;
                grid-template-columns: auto 1fr;
                margin: 0;
                padding: 0;
            }
        </style>
        <script type="text/javascript" src="https://cdn.bokeh.org/bokeh/release/bokeh-3.2.2.min.js"></script>
        <script type="text/javascript" src="https://cdn.bokeh.org/bokeh/release/bokeh-gl-3.2.2.min.js"></script>
        <script type="text/javascript" src="https://cdn.bokeh.org/bokeh/release/bokeh-widgets-3.2.2.min.js"></script>
        <script type="text/javascript">
            Bokeh.set_log_level("info");
        </script>
        <div class="bokeh_scatter" id=@id>
            @div
            @osd
        </div>
        @script
    '''
    html = html.replace('@id', id)
    html = html.replace('@div', div)
    html = html.replace('@osd', osd)
    html = html.replace('@script', script).replace('<script ', '<script eval ')

    return html

def check_colormap_type(cmap_name):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as colors

    # Check if the colormap name is valid
    if cmap_name not in plt.colormaps():
        raise ValueError(f"{cmap_name} is not a valid Matplotlib colormap.")

    # Retrieve the colormap object
    cmap = cm.get_cmap(cmap_name)

    # Check if the colormap is continuous or discrete
    if isinstance(cmap, colors.LinearSegmentedColormap):
        return 'continuous'
    elif isinstance(cmap, colors.ListedColormap):
        return 'discrete'
    else:
        raise ValueError(f"{cmap_name} type is unknown.")

def bokeh_scatter(
    df,
    x,
    y,
    hues: Sequence[str],
    hover_columns: Sequence[str] | None = None,
    title='Scatter plot',
    invert: str = '',
    marker: Literal['circle', 'square'] = 'circle',
    size: float | int = 10,
    aspect_ratio: float | None = None,
):
    if not hues:
        raise ValueError('At least one hue is needed')

    v = 4

    cols = [x, y, *(hover_columns or []), *hues]
    for c in 'barcode site well overlay overlay_style'.split():
        if c in df.columns:
            cols += [c]

    cols = list(set(cols))
    df = df[cols]

    source = ColumnDataSource(data=df)

    # Mappers and color bars
    mappers = {}
    color_bars = {}

    for hue in hues:
        linear = (
            np.issubdtype(df[hue], np.floating)
            or np.issubdtype(df[hue], np.integer)
            and df[hue].nunique() > 10
        )
        if linear:
            mappers[hue] = LinearColorMapper(
                palette="Viridis256",
                low=df[hue].min(),
                high=df[hue].max(),
            )
        else:
            factors = df[hue].unique()
            mappers[hue] = CategoricalColorMapper(
                palette=Category10[10][: len(factors)],
                factors=factors,
            )
        color_bars[hue] = ColorBar(
            color_mapper=mappers[hue],
            height=300 if linear else 24 * len(factors),
            width=15 if linear else 24,
            background_fill_color='rgba(0,0,0,0)',
            title=hue,
        )

    if aspect_ratio is not None:
        aspect = dict(
            match_aspect=True,
            aspect_scale=aspect_ratio,
        )
    else:
        aspect = {}

    p = figure(
        width=1000,
        height=1000,
        x_axis_label=x,
        y_axis_label=y,
        x_range=DataRange1d(
            flipped='x' in invert, max_interval=1.2 * max(np.ptp(df[x]), np.ptp(df[y]))
        ),
        y_range=DataRange1d(
            flipped='y' in invert, max_interval=1.2 * max(np.ptp(df[x]), np.ptp(df[y]))
        ),
        title=title,
        tools='lasso_select box_select wheel_zoom box_zoom reset fullscreen pan'.split(),
        # toolbar_location='below',
        toolbar_location='right',
        active_scroll='wheel_zoom',
        active_drag='box_select',
        border_fill_color='#2d2d2d',
        output_backend='webgl',
        **aspect,
    )

    if hover_columns:
        hover = HoverTool(
            tooltips=[(c, f'@{c}') for c in hover_columns],
            mode='mouse',
            point_policy='snap_to_data',
        )
        p.add_tools(hover)

    p2 = figure(
        width=150,
        height=p.height,
        tools='',
        toolbar_location=None,
        outline_line_alpha=0,
        background_fill_color='rgba(0,0,0,0)',
        border_fill_color='rgba(0,0,0,0)',
    )
    if marker == 'square':
        scatter = p.rect(
            x=x,
            y=y,
            width=size, height=size,
            source=source,
            fill_color={
                'field': hues[0],
                'transform': mappers[hues[0]],
            },
            line_color='#111',
            alpha=0.9,
            line_alpha=1.0,
            line_width=1.0,
        )
        # Add invisible circles for lasso selection
        p.circle(x=x, y=y, size=10, source=source, color=None, alpha=0)
    elif marker == 'circle':
        scatter = p.circle(
            x=x,
            y=y,
            size=size,
            source=source,
            fill_color={
                'field': hues[0],
                'transform': mappers[hues[0]],
            },
            line_color='#111',
            alpha=0.9,
            line_alpha=1.0,
            line_width=1.0,
        )
        # Add invisible circles for lasso selection
        # p.circle(x=x, y=y, size=10, source=source, color=None, alpha=0)
    else:
        raise ValueError(f'Unsupported {marker=}!')

    scatter.selection_glyph = type(scatter.glyph)(**scatter.glyph.properties_with_values())
    scatter.nonselection_glyph = type(scatter.glyph)(**scatter.glyph.properties_with_values())
    scatter.selection_glyph.line_color = '#eee'
    scatter.nonselection_glyph.fill_alpha = 0.7
    scatter.nonselection_glyph.line_alpha = 0.9
    for hue in hues:
        p2.add_layout(color_bars[hue], 'right')
        color_bars[hue].visible = False
    color_bars[hues[0]].visible = True

    # Callback
    js_code = """
        scatter.glyph.fill_color = {field: hue, transform: mappers[hue]}
        scatter.selection_glyph.fill_color = {field: hue, transform: mappers[hue]};
        scatter.nonselection_glyph.fill_color = {field: hue, transform: mappers[hue]};
        for (let hue_i of Object.keys(color_bars)) {
            color_bars[hue_i].visible = hue_i === hue
        }
        source.change.emit()
    """

    # Buttons
    buttons = []
    for hue in hues:
        button = Button(label=hue, button_type='success')
        callback = CustomJS(
            args=dict(
                source=source,
                scatter=scatter,
                mappers=mappers,
                color_bars=color_bars,
                hue=hue,
            ),
            code=js_code,
        )
        button.js_on_click(callback)
        buttons += [button]

    if 0:
        # experiment to change x y dynamically
        for coords in 'xy xz yz'.split():
            button = Button(label=coords, button_type='success')
            callback = CustomJS(
                args=dict(
                    p=p,
                    source=source,
                    scatter=scatter,
                    coords=coords,
                    x_axis=p.xaxis[0],
                    y_axis=p.yaxis[0],
                ),
                code='''
                    scatter.glyph.x.field = coords[0]
                    scatter.glyph.y.field = coords[1]
                    x_axis.axis_label = coords[0]
                    y_axis.axis_label = coords[1]
                    source.change.emit()
                    p.change.emit()
                ''',
            )
            button.js_on_click(callback)
            buttons += [button]

    p.js_on_event(
        'doubletap',
        CustomJS(
            args=dict(source=source),
            code="""
                source.selected.indices = []
                source.change.emit()
            """,
        ),
    )

    id = f'scatter-{uuid.uuid4()}'

    source.selected.js_on_change('indices', CustomJS(
        args=dict(source=source, id=id),
        code="""
            const data = source.data
            const rows = []
            for (const i of source.selected.indices) {
              const row = {}
              for (const col of 'barcode well site clip'.split(' ')) {
                if (data.hasOwnProperty(col)) 
                  row[col] = data[col][i]
              }
              rows.push(row)
            }
            console.log(rows)
            const osd_iframe = document.querySelector(`#${id} iframe`)

            function call_osd(method, ...args) {
              osd_iframe.contentWindow.postMessage({method, arguments: args}, '*')
            }
            call_osd('highlight_tile', {barcode, well, site})
        """,
    ))

    # Layout
    l = layout([buttons, [p2, p]])

    return bokeh_to_html(l, id=id)




def bokeh_scatter_with_selection(
    df,
    x,
    y,
    hues: Sequence[str],
    hover_columns: Sequence[str] | None = None,
    title='Scatter plot',
    invert: str = '',
    marker: Literal['circle', 'square'] = 'circle',
    size: float | int = 10,
    aspect_ratio: float | None = None,
    filter_column: str = None
):
    if not hues:
        raise ValueError('At least one hue is needed')

    v = 4

    cols = [x, y, filter_column, *(hover_columns or []), *hues]
    for c in 'barcode site well overlay overlay_style'.split():
        if c in df.columns:
            cols += [c]

    cols = list(set(cols))
    df = df[cols]

    source = ColumnDataSource(data=df)
    original_source = ColumnDataSource(data=df.copy())

    mappers = {}
    color_bars = {}

    for hue in hues:
        linear = (
            np.issubdtype(df[hue], np.floating)
            or np.issubdtype(df[hue], np.integer)
            and df[hue].nunique() > 10
        )
        if linear:
            mappers[hue] = LinearColorMapper(
                palette="Viridis256",
                low=df[hue].min(),
                high=df[hue].max(),
            )
        else:
            factors = df[hue].unique()
            mappers[hue] = CategoricalColorMapper(
                palette=Category10[10][: len(factors)],
                factors=factors,
            )
        color_bars[hue] = ColorBar(
            color_mapper=mappers[hue],
            height=300 if linear else 24 * len(factors),
            width=15 if linear else 24,
            background_fill_color='rgba(0,0,0,0)',
            title=hue,
        )

    if aspect_ratio is not None:
        aspect = dict(
            match_aspect=True,
            aspect_scale=aspect_ratio,
        )
    else:
        aspect = {}

    p = figure(
        width=1000,
        height=1000,
        x_axis_label=x,
        y_axis_label=y,
        x_range=DataRange1d(
            flipped='x' in invert, max_interval=1.2 * max(np.ptp(df[x]), np.ptp(df[y]))
        ),
        y_range=DataRange1d(
            flipped='y' in invert, max_interval=1.2 * max(np.ptp(df[x]), np.ptp(df[y]))
        ),
        title=title,
        tools='lasso_select box_select wheel_zoom box_zoom reset fullscreen pan'.split(),
        # toolbar_location='below',
        toolbar_location='right',
        active_scroll='wheel_zoom',
        active_drag='box_select',
        border_fill_color='#2d2d2d',
        output_backend='webgl',
        **aspect,
    )

    if hover_columns:
        hover = HoverTool(
            tooltips=[(c, f'@{c}') for c in hover_columns],
            mode='mouse',
            point_policy='snap_to_data',
        )
        p.add_tools(hover)

    p2 = figure(
        width=150,
        height=p.height,
        tools='',
        toolbar_location=None,
        outline_line_alpha=0,
        background_fill_color='rgba(0,0,0,0)',
        border_fill_color='rgba(0,0,0,0)',
    )
    if marker == 'square':
        scatter = p.rect(
            x=x,
            y=y,
            width=size, height=size,
            source=source,
            fill_color={
                'field': hues[0],
                'transform': mappers[hues[0]],
            },
            line_color='#111',
            alpha=0.9,
            line_alpha=1.0,
            line_width=1.0,
        )
        # Add invisible circles for lasso selection
        p.circle(x=x, y=y, size=10, source=source, color=None, alpha=0)
    elif marker == 'circle':
        scatter = p.circle(
            x=x,
            y=y,
            size=size,
            source=source,
            fill_color={
                'field': hues[0],
                'transform': mappers[hues[0]],
            },
            line_color='#111',
            alpha=0.9,
            line_alpha=1.0,
            line_width=1.0,
        )
        # Add invisible circles for lasso selection
        # p.circle(x=x, y=y, size=10, source=source, color=None, alpha=0)
    else:
        raise ValueError(f'Unsupported {marker=}!')

    scatter.selection_glyph = type(scatter.glyph)(**scatter.glyph.properties_with_values())
    scatter.nonselection_glyph = type(scatter.glyph)(**scatter.glyph.properties_with_values())
    scatter.selection_glyph.line_color = '#eee'
    scatter.nonselection_glyph.fill_alpha = 0.7
    scatter.nonselection_glyph.line_alpha = 0.9
    for hue in hues:
        p2.add_layout(color_bars[hue], 'right')
        color_bars[hue].visible = False
    color_bars[hues[0]].visible = True

    # Callback
    js_code = """
        scatter.glyph.fill_color = {field: hue, transform: mappers[hue]}
        scatter.selection_glyph.fill_color = {field: hue, transform: mappers[hue]};
        scatter.nonselection_glyph.fill_color = {field: hue, transform: mappers[hue]};
        for (let hue_i of Object.keys(color_bars)) {
            color_bars[hue_i].visible = hue_i === hue
        }
        source.change.emit()
    """

    # Buttons
    buttons = []
    for hue in hues:
        button = Button(label=hue, button_type='success')
        callback = CustomJS(
            args=dict(
                source=source,
                scatter=scatter,
                mappers=mappers,
                color_bars=color_bars,
                hue=hue,
            ),
            code=js_code,
        )
        button.js_on_click(callback)
        buttons += [button]

    if 0:
        # experiment to change x y dynamically
        for coords in 'xy xz yz'.split():
            button = Button(label=coords, button_type='success')
            callback = CustomJS(
                args=dict(
                    p=p,
                    source=source,
                    scatter=scatter,
                    coords=coords,
                    x_axis=p.xaxis[0],
                    y_axis=p.yaxis[0],
                ),
                code='''
                    scatter.glyph.x.field = coords[0]
                    scatter.glyph.y.field = coords[1]
                    x_axis.axis_label = coords[0]
                    y_axis.axis_label = coords[1]
                    source.change.emit()
                    p.change.emit()
                ''',
            )
            button.js_on_click(callback)
            buttons += [button]

    p.js_on_event(
        'doubletap',
        CustomJS(
            args=dict(source=source),
            code="""
                source.selected.indices = []
                source.change.emit()
            """,
        ),
    )

    id = f'scatter-{uuid.uuid4()}'

    source.selected.js_on_change('indices', CustomJS(
        args=dict(source=source, id=id),
        code="""
            const data = source.data
            const rows = []
            for (const i of source.selected.indices) {
              const row = {}
              for (const col of 'barcode well site clip'.split(' ')) {
                if (data.hasOwnProperty(col)) 
                  row[col] = data[col][i]
              }
              rows.push(row)
            }
            console.log(rows)
            const osd_iframe = document.querySelector(`#${id} iframe`)

            function call_osd(method, ...args) {
              osd_iframe.contentWindow.postMessage({method, arguments: args}, '*')
            }
            call_osd('update_tile_source', rows)
        """,
    ))

    if filter_column is not None:
        unique_values = sorted(df[filter_column].unique().tolist())
        filter_widget = CheckboxGroup(labels=unique_values, active=list(range(len(unique_values))))  # All items selected by default
        filter_callback = CustomJS(args=dict(source=source, original_source=original_source, checkbox_group=filter_widget, labels=unique_values), code="""
            const selected_indices = checkbox_group.active;
            const selected_labels = selected_indices.map(index => labels[index]);
            const data = source.data;
            const original_data = original_source.data;
            
            // Reset data
            for (const key in data) {
                data[key] = [];
            }
            
            // Filter data based on selected labels
            for (let i = 0; i < original_data['index'].length; ++i) {
                if (selected_labels.includes(original_data['"""+filter_column+"""'][i])) {
                    for (let key in data) {
                        data[key].push(original_data[key][i]);
                    }
                }
            }
            
            source.change.emit();
        """)
        filter_widget.js_on_change('active', filter_callback)
        #filter_widget.js_on_change('active', filter_callback)
        
        #filter_widget.js_on_change('value', callback)

    else:
        filter_widget = None

    buttons_layout = row(buttons)  # Assuming 'buttons' is a list of Button widgets
    plot_layout = row(p2, p)  # Your original plot arrangement

    # Updated layout with filter widget
    if filter_widget is not None:
        l = column(filter_widget, buttons_layout, plot_layout)
    else:
        l = column(buttons_layout, plot_layout)

    return bokeh_to_html(l, id=id)




def bokeh_scatter_with_selection_and_filters(
    df,
    x,
    y,
    hues,
    filter_columns,
    hover_columns=None,
    title='Scatter plot',
    marker='circle',
    size=10,
    aspect_ratio=None,
):
    if not hues:
        raise ValueError('At least one hue is needed')
    
    # Preparing the data source and figure
    source = ColumnDataSource(data=df)
    p = figure(title=title, tools='lasso_select,box_select,wheel_zoom,reset', sizing_mode='stretch_width', height=400)
    color_mapper = linear_cmap(field_name=hues[0], palette=Viridis256, low=min(df[hues[0]]), high=max(df[hues[0]]))
    
    # Drawing the scatter plot
    if marker == 'circle':
        scatter = p.circle(x=x, y=y, source=source, size=size, color=color_mapper, line_color='black')
    elif marker == 'square':
        scatter = p.square(x=x, y=y, source=source, size=size, color=color_mapper, line_color='black')
    else:
        raise ValueError(f"Unsupported marker type: {marker}")
    
    # Widgets and callbacks for dynamic filtering
    widgets = []
    for col in filter_columns:
        unique_values = ['All'] + sorted(df[col].unique().tolist())
        widget = Select(title=col, value='All', options=unique_values)
        widgets.append(widget)
    
    # JavaScript to filter data based on widget selections
    combined_filter_js = """
    const data = source.data;
    const original_data = source.data; // Assuming there's a way to access the original unfiltered data
    let indices_to_show = [];
    for (let i = 0; i < original_data['index'].length; i++) {
        let include = true;
        %s
        if (include) {
            indices_to_show.push(i);
        }
    }
    Object.keys(data).forEach(key => {
        data[key] = indices_to_show.map(index => original_data[key][index]);
    });
    source.change.emit();
    """
    
    filter_checks = ""
    for widget in widgets:
        filter_checks += f"""
        if (cb_obj.title === '{widget.title}' && cb_obj.value !== 'All') {{
            include = include && (original_data['{widget.title}'][i] === cb_obj.value);
        }}
        """
    combined_filter_js = combined_filter_js % filter_checks
    
    # Attach callback to widgets
    for widget in widgets:
        widget.js_on_change('value', CustomJS(args=dict(source=source), code=combined_filter_js))
    
    # Layout
    layout = column(*widgets, p)
    
    return layout
