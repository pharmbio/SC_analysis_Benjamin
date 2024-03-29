from autogluon.tabular import TabularPredictor
import click
import bokeh
print(bokeh.__version__)

@click.command()
@click.option('input_dataset','-i', type=click.Path(exists=True))

def main(input_dataset):
    autogluon_res = TabularPredictor.load(input_dataset)
    autogluon_res.fit_summary(show_plot = True)


if __name__ == '__main__':
    main()