
'''
This is to download raw data from Google Drive
It requires gdown and rarfile packages
as well as unrar should be installed
'''

import gdown
import rarfile
import os
from tqdm import tqdm

def download_and_extract_rar(drive_link, output_folder):
    # Convert the Google Drive shareable link to a downloadable link
    file_id = drive_link.split('/d/')[1].split('/')[0]
    download_url = f'https://drive.google.com/uc?id={file_id}'
    
    # Set the output rar file name
    # rar_file_name = os.path.join(output_folder, 'downloaded_file.rar')

    # Download the rar file
    print("Downloading file...")
    info = gdown.download(download_url, quiet=False, fuzzy=True)
    
    if info is None:
        print("Failed to download the file.")
        return

    file_name = info
    print(f"File downloaded to: {file_name}")
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Extract the rar file
    if file_name.endswith('.rar'):
        rar_base_name = os.path.splitext(file_name)[0]
        extraction_path = os.path.join(output_folder, rar_base_name)
        
        if not os.path.exists(extraction_path):
            os.makedirs(extraction_path)

        # Extract rar file into the new subfolder
        print(f"Extracting files to {extraction_path}...")
        with rarfile.RarFile(file_name, 'r') as rar_ref:
            # Initialize tqdm for progress bar
            total_files = len(rar_ref.infolist())
            with tqdm(total=total_files, desc="Extracting", unit="file") as pbar:
                for file_info in rar_ref.infolist():
                    rar_ref.extract(file_info, extraction_path)
                    pbar.update(1)
        
        print(f"Files extracted to: {extraction_path}")
    else:
        print(f"The downloaded file '{file_name}' is not a RAR file.")

    
    
    # Clean up the downloaded rar file
    print("Cleaning up...")
    os.remove(file_name)
    print("Cleanup complete. rar file removed.")

def main():
    drive_link = 'https://drive.google.com/file/d/1LTyQhC9HOumWf8gDJY56Dp4cgg_V0ojb/view?usp=sharing' 
    output_folder = 'datasets/raw_data'

    download_and_extract_rar(drive_link, output_folder)

    return

if __name__ == "__main__":
    main()