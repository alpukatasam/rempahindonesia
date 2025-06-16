def download_model_from_gdrive(file_id, output_path):
    url = f'https://drive.google.com/uc?id={file_id}'
    response = requests.get(url, stream=True)
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)