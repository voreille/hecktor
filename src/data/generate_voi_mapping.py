from pathlib import Path
import json

project_dir = Path(__file__).resolve().parents[2]
data_dir = project_dir / "data/hecktor2022/processed/"
center = "chb"
labels_dir = data_dir / f"{center}/images"
output_file = data_dir / f"{center}/vois_mapping.json"

if "chup" in center.lower():
    gtvt_labels = [
        'GTV(2)', 'RGTVt', 'GTV_HEX', 'ROI-1', 'ROI-2', 'GTVp', 'ROI-3',
        'GTV_HE', 'GTVt', 'GTV', 'GTVt(2)'
    ]
    gtvn_labels = [
        'GTVn15', 'GTVn08', 'GTVn13', 'GTVn10', 'GTVn', 'GVn01', 'GTVn12',
        'GTVn06', 'GTVn11', 'GTVn07', 'GTVn14', 'GTVn02', 'GTVn09', 'GTVn03',
        'GTVn16', 'GTVn01', 'GTVn04', 'GTVn05'
    ]
elif "mda_test" in center.lower():
    gtvt_labels = [
        'GTV-P', 'GTVP', 'GTV_P', 'GTVp', 'GTVp-YK', 'GTVp2', 'GTVp_KW',
        'GTVp_KW_SA', 'GTVp_SA', 'GTVp_YK', 'GTVp_YK_SA'
    ]
    gtvn_labels = [
        'GTNn2', 'GTV_7', 'GTV_N1', 'GTV_N10', 'GTV_N11', 'GTV_N2', 'GTV_N3',
        'GTV_N4', 'GTV_N5', 'GTV_N6', 'GTV_N7', 'GTV_N8', 'GTV_N9', 'GTV_N_1',
        'GTV_N_2', 'GTVn01', 'GTVn1', 'GTVn10', 'GTVn11', 'GTVn12', 'GTVn13',
        'GTVn14', 'GTVn15', 'GTVn1_SA', 'GTVn2', 'GTVn2_SA', 'GTVn3',
        'GTVn3_SA', 'GTVn4', 'GTVn4_SA', 'GTVn5', 'GTVn5_SA', 'GTVn6',
        'GTVn6_SA', 'GTVn7', 'GTVn8', 'GTVn9'
    ]

elif "mda_train" in center.lower():
    gtvn_labels = ["GTVn"]
    gtvt_labels = ["GTVp"]
elif "montreal" in center.lower():
    gtvn_labels = [
        'GTV_N', 'GTVn', 'GTVn(2)', 'GTVn01', 'GTVn02', 'GTVn03', 'GTVn04',
        'GTVn05', 'GTVn06', 'GTVn07', 'GTVn08', 'GTVn1', 'GTVn2'
    ]
    gtvt_labels = ["GTVt"]
elif "chb" in center.lower():
    gtvn_labels = [
        'GTBn04', 'GTVn01', 'GTVn02', 'GTVn03', 'GTVn04', 'GTVn05', 'GTVn06',
        'GTVn07', 'GTVn08', 'GTVn09', 'GTVn10', 'GTVn11', 'GTVn12', 'GTVn13',
        'GTVn14', 'GTVn15', 'GTvn08', 'GVTn01', 'GVTn02', 'GVTn03'
    ]
    gtvt_labels = ['GTVt01', 'GTVt02', 'GVTt01']
else:
    gtvn_labels = ["GTVn"]
    gtvt_labels = ["GTVt"]


def main():
    files = list(labels_dir.glob("*RTSTRUCT*.nii.gz"))
    patient_ids = list(set([f.name.split("__")[0] for f in files]))
    if "chup" in center.lower():
        patient_ids.sort(key=lambda x: int(x[2:]))
    if "chuv" in center.lower():
        patient_ids.sort(
            key=lambda x: int(x.strip("HNCHUV").strip("P").strip("_ORL")))
    if "montreal" in center.lower():
        patient_ids.sort(
            key=lambda x: (str(x.split("-")[1]), int(x.split("-")[-1])))
    elif "mda_test" in center.lower():
        patient_ids.sort(key=lambda x: int(x))
    elif "mda_train" in center.lower():
        patient_ids.sort(key=lambda x: int(x.split("-")[2]))
    elif "chb" in center.lower():
        patient_ids.sort(key=lambda x: int(x[3:]))
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
