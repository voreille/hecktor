from pathlib import Path
import json

project_dir = Path(__file__).resolve().parents[2]
data_dir = project_dir / "data/hecktor2022/processed/"
center = "CHUV"
labels_dir = data_dir / f"{center}/images"
output_file = data_dir / f"{center}/vois_mapping.json"

if "chup" in center.lower():
    gtvt_labels = [
        'GTV(2)', 'RGTVt', 'GTV_HEX', 'ROI-1', 'ROI-2', 'GTVp', 'ROI-3',
        'GTV_HE', 'GTVt', 'GTV', 'GTVt(2)'
    ]
else:
    gtvt_labels = ["GTVt"]

if "chup" in center.lower():
    gtvn_labels = [
        'GTVn15', 'GTVn08', 'GTVn13', 'GTVn10', 'GTVn', 'GVn01', 'GTVn12',
        'GTVn06', 'GTVn11', 'GTVn07', 'GTVn14', 'GTVn02', 'GTVn09', 'GTVn03',
        'GTVn16', 'GTVn01', 'GTVn04', 'GTVn05'
    ]
elif "montreal" in center.lower():
    gtvn_labels = [
        'GTV_N', 'GTVn', 'GTVn(2)', 'GTVn01', 'GTVn02', 'GTVn03', 'GTVn04',
        'GTVn05', 'GTVn06', 'GTVn07', 'GTVn08', 'GTVn1', 'GTVn2'
    ]
else:
    gtvn_labels = [
        'GTV_N', 'GTVn', 'GTVn01', 'GTVn02', 'GTVn03', 'GTVn04', 'GTVn1',
        'GTVn2'
    ]


def main():
    files = list(labels_dir.glob("*RTSTRUCT*.nii.gz"))
    patient_ids = list(set([f.name.split("__")[0] for f in files]))
    if "chup" in center.lower():
        patient_ids.sort(key=lambda x: int(x[2:]))
    if "chuv" in center.lower():
        patient_ids.sort(key=lambda x: int(x.strip("HNCHUV").strip("P").strip("_ORL")))
    if "montreal" in center.lower():
        patient_ids.sort(
            key=lambda x: (str(x.split("-")[1]), int(x.split("-")[-1])))
    results = {}
    for patient in patient_ids:
        vois = [
            f.name.split("__")[1] for f in labels_dir.rglob(f"{patient}__*")
        ]

        results.update({
            patient: {
                "GTVt":
                list(set([label for label in vois if label in gtvt_labels])),
                "GTVn":
                list(set([label for label in vois if label in gtvn_labels])),
            }
        })
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    main()
