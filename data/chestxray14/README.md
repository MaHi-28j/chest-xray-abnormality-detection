# NIH ChestX-ray14 Raw Data Directory

Place the NIH ChestX-ray14 files here:
- `Data_Entry_2017.csv`: The official dataset metadata table.
- `images/`: Directory containing the chest radiograph PNG files.

### Automatic Download
You do not need to download these files manually. Run:
```bash
python src/setup_dataset.py
```
The script will download the metadata and images automatically.

### Manual Download (Optional)
If preferred, you can download files directly from the official NIH Box repository:
https://nihcc.app.box.com/v/ChestXray-NIHCC
and extract the PNG images into `data/chestxray14/images/`.
