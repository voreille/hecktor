from pathlib import Path
import json

project_dir = Path(__file__).resolve().parents[2]

nii_folder = project_dir / "data/interobserver/nii"


def main(labels_dict):

    for observer_dir in nii_folder.iterdir():
        observer = observer_dir.name
        labels_dir = project_dir / f"data/interobserver/nii/{observer}/images"
        output_file = project_dir / f"data/interobserver/vois_mapping_{observer}.json"

        files = list(labels_dir.glob("*RTSTRUCT*.nii.gz"))
        patient_ids = list(set([f.name.split("__")[0] for f in files]))
        print(set([f.name.split('__')[1] for f in files]))
        gtvt_labels = labels_dict[observer]["GTVt"]
        gtvn_labels = labels_dict[observer]["GTVn"]

        results = {}
        for patient in patient_ids:
            vois = [
                f.name.split("__")[1]
                for f in labels_dir.rglob(f"{patient}__*")
            ]

            results.update({
                patient: {
                    "GTVt":
                    list(set([label for label in vois
                              if label in gtvt_labels])),
                    "GTVn":
                    list(set([label for label in vois
                              if label in gtvn_labels])),
                }
            })
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == '__main__':
    gtv_labels_dict = {
        "leo": {
            "GTVt": [
                'GTVt01',
                'GTVt1',
            ],
            "GTVn": [
                'GTVn01',
                'GTVn05',
                'GTVn03',
                'GTVn04',
                'GTVn02',
                'GTVn06',
                'GTvn01',
            ],
        },
        "dina": {
            "GTVt": [
                'GTV_P',
            ],
            "GTVn": [
                'GTV_N',
                'GTV_N1',
                'GTV_N2',
            ],
        },
        "mario": {
            "GTVt": [
                'GTVt',
                'GTVp',
            ],
            "GTVn": [
                'GTVn1',
                'GTVn2',
                'GTVn',
                'ROI-2',
            ],
        },
        "moamen": {
            "GTVt": [
                'GTVP',
            ],
            "GTVn": [
                'GTVn1',
                'GTVN4',
                'GTVN6',
                'GTVN2',
                'GTVN3',
                'GTVN5',
                'GTVN1',
            ],
        },
        "olena": {
            "GTVt": [
                'GTVp',
            ],
            "GTVn": [
                'GTVn07',
                'GTVn06',
                'GTVn03',
                'GTVn02',
                'GTVn04',
                'GTVn05',
                'GTVn',
                'GTVn01',
            ],
        },
        "panagiotis": {
            "GTVt": [
                'GTVp',
            ],
            "GTVn": [
                'GTVn',
            ],
        },
        "ricardo": {
            "GTVt": [
                'GVTt',
                'GTVt',
            ],
            "GTVn": [
                'GTVn01',
                'GTVn',
                'GTVn03',
                'GTVn02',
                'GTVt02',
                'GTVn04',
            ],
        },
        "sarah": {
            "GTVt": [
                'GTV_T',
                'GTV_1',
                'GTV_2',
            ],
            "GTVn": [
                'GVT_N2',
                'GTV_N5',
                'GTV_N4',
                'GTV_N2',
                'GTV_N',
                'GTV_N1',
                'GTV_N3',
            ],
        },
        "yomna": {
            "GTVt": [
                'GTVp',
                'GTVn3',
                'GTVn1',
            ],
            "GTVn": [
                'GTVn2',
            ],
        },
    }

    main(gtv_labels_dict)
