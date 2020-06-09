import os
import glob
from shutil import copyfile

import click

path_in = '../../data/segmentation_output'
path_out = '../../data/segmentation_output_renamed'


@click.command()
@click.argument('input_folder', type=click.Path(exists=True), default=path_in)
@click.argument('output_folder', type=click.Path(), default=path_out)
def main(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    for infile in glob.glob(os.path.join(input_folder, '*.gz')):
        # Copy with correct name for evaluation
        outfile = os.path.join(output_folder,
                               (os.path.basename(infile)).split('_')[2] +
                               '.nii.gz')
        copyfile(infile, outfile)


if __name__ == '__main__':
    main()
