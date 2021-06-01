from pathlib import Path
from datetime import datetime
from shutil import move

import click
import pydicom as pdcm
from tqdm import tqdm

project_dir = Path(__file__).resolve().parents[2]
default_input_folder = project_dir / "data/raw/poitier_train"
default_archive_folder = project_dir / "data/surnumerary_rtstruct/"


@click.command()
@click.argument('input_folder',
                type=click.Path(exists=True),
                default=default_input_folder)
@click.argument('archive_folder',
                type=click.Path(),
                default=default_archive_folder)
def main(input_folder, archive_folder):
    rtstruct_paths = [f for f in Path(input_folder).rglob("*RTSTRUCT*.dcm")]
    rt_data_list = [
        pdcm.read_file(f, stop_before_pixels=True)
        for f in tqdm(rtstruct_paths)
    ]
    rt_list = list(zip(rtstruct_paths, rt_data_list))
    patient_names = [f.PatientName for f in rt_data_list]
    patient_names = list(set(patient_names))
    archive_folder = Path(archive_folder)
    archive_folder.mkdir(parents=True, exist_ok=True)
    for patient_name in patient_names:
        rt_list_p = [f for f in rt_list if f[1].PatientName == patient_name]
        rt_list_p.sort(key=lambda x: get_datetime(x[1]))
        rts_to_move = rt_list_p[:-1]
        path_to_move = Path(archive_folder) / str(patient_name)
        path_to_move.mkdir(exist_ok=True)
        for r in rts_to_move:
            file_source = r[0]
            file_destination = path_to_move / file_source.name
            new_path = file_destination
            counter = 0
            while new_path.is_file():
                counter += 1
                new_path = file_destination.with_name(
                    file_destination.name.replace(file_destination.suffix, '')
                    + '(' + str(counter) + ')' + file_destination.suffix)

            move(str(file_source.resolve()), str(new_path.resolve()))


def get_datetime(s):
    return datetime.strptime(s.SeriesDate + s.SeriesTime.split('.')[0],
                             "%Y%m%d%H%M%S")


if __name__ == '__main__':
    main()
