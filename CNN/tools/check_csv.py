import os
import pandas as pd
from PIL import Image
from tqdm import tqdm  # Import tqdm for the progress bar

def check_integrity(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img.verify()
        return True
    except:
        os.remove(image_path)
        return False

def check_files_in_folder(csv_file, data_dir):
    # Carica il CSV in un DataFrame
    data_df = pd.read_csv(csv_file)

    # Verifica se i file sono presenti nella cartella con progress bar
    for index, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Checking files"):
        file_name = row.iloc[0]  # Legge il nome del file dalla prima colonna
        file_path = os.path.join(data_dir, file_name)

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            # Rimuovi la riga dal DataFrame
            data_df.drop(index, inplace=True)

        if check_integrity(file_path) is False:
            print(f"File Corrupted: {file_path}")
            # Rimuovi la riga dal DataFrame
            data_df.drop(index, inplace=True)
            

    # Salva il DataFrame aggiornato in un nuovo file CSV
    new_csv_file = "updated_" + os.path.basename(csv_file)
    data_df.to_csv(new_csv_file, index=False)
    print(f"Updated CSV saved to: {new_csv_file}")

if __name__ == "__main__":
    csv_file_path = '../data/combined_annotations.csv'
    data_folder = '../data/images/images'

    check_files_in_folder(csv_file_path, data_folder)
