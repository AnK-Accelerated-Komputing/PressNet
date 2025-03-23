'''
    this is to download multiple .h5 and .json files from google drive link
'''

import gdown
import os

def download_file(drive_link, output_folder):
    # Convert the Google Drive shareable link to a downloadable link
    file_id = drive_link.split('/d/')[1].split('/')[0]
    download_url = f'https://drive.google.com/uc?id={file_id}'
    
    # Download the file
    print("Downloading file...")
    file_name = gdown.download(download_url, quiet=False, fuzzy=True)
    
    if file_name is None:
        print("Failed to download the file.")
        return
    
    print(f"File downloaded: {file_name}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Move file to the output folder
    destination_path = os.path.join(output_folder, os.path.basename(file_name))
    os.rename(file_name, destination_path)
    print(f"File saved to: {destination_path}")

def main():
    drive_links = [
        'https://drive.google.com/file/d/1q-jIcjpqsBeurIl-MN1EvOQBM6uEFoqs/view?usp=sharing',  
        'https://drive.google.com/file/d/1RuUy03JvPs02GuVFLiZApmD_uiZB1Kwr/view?usp=sharing'
    ]
    output_folder = '/home/user/PressNet/surrogateAI/data'
    
    for link in drive_links:
        download_file(link, output_folder)

if __name__ == "__main__":
    main()
